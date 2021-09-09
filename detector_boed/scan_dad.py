""""""


import warnings
import numpy as np
import pandas as pd

import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.infer.util import torch_item

from tqdm import trange

from neural.modules import SetEquivariantDesignNetwork

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimation

from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117

warnings.filterwarnings("ignore")


class EncoderNetwork(nn.Module):
    """Encoder network for location finding example"""

    def __init__(self, design_dim, osbervation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        input_dim = self.design_dim_flat + osbervation_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, xi, y, **kwargs):
        xi = xi.flatten(-2)
        inputs = torch.cat([xi, y], dim=-1).float()
        # inputs = torch.cat([xi, y], dim=-1)
        print("inputs shape ", inputs.shape)
        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    """Emitter network for location finding example"""

    def __init__(self, encoding_dim, design_dim):
        super().__init__()
        self.design_dim = design_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        self.linear = nn.Linear(encoding_dim, self.design_dim_flat)

    def forward(self, r):
        xi_flat = self.linear(r)
        return xi_flat.reshape(xi_flat.shape[:-1] + self.design_dim)


class ScanOptDAD(nn.Module):
    """Location finding example"""

    def __init__(
        self,
        scan_map_class: ScanMapClass,
        design_net,
        base_signal=0.1,  # G-map hyperparam
        max_signal=1e-4,  # G-map hyperparam
        theta_loc=None,  # prior on theta mean hyperparam
        theta_covmat=None,  # prior on theta covariance hyperparam
        noise_scale=None,  # this is the scale of the noise term
        weight_dim: int = 4,  # physical dimension
        K=1,  # number of sources
        T=2,  # number of experiments
        detector: int = 0,
    ):
        super().__init__()
        self._w_dim = weight_dim
        self._smc = scan_map_class
        self._detector = detector

        self.design_net = design_net
        self.base_signal = base_signal
        self.max_signal = max_signal
        # Set prior:
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, weight_dim))
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(weight_dim)
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)
        # Observations noise scale:
        self.noise_scale = noise_scale if noise_scale is not None else torch.tensor(1.0)
        self.n = 1  # batch=1
        self.p = weight_dim  # dimension of theta (location finding example will be 1, 2 or 3).
        self.T = T  # number of experiments

        self.sig = torch.nn.Sigmoid()

    def forward_map(self, x_new, theta):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.
        """

        y = []
        x_resh = 0.5 * (x_new + theta)
        x_resh = torch.flatten(x_resh, 0, len(x_new.shape) - 3)
        for x in x_resh:
            n_new = self.sig(x)
            n_new = torch.flatten(n_new).detach().cpu().numpy() * 0.2 + 0.9

            new_weights = self.construct_weights(n_new, self._detector)
            self._smc.calc_peak_positions(weights=np.array(new_weights))
            losses = self._smc.calc_loss()
            y_out = torch.tensor([losses[1]], requires_grad=True)
            y.append(y_out)
        y_out = torch.reshape(torch.tensor(y, requires_grad=True), (x_new.shape[:-1]))

        print("shapes: ", x_new.shape, y_out.shape, " , loss: ", y_out)

        return y_out

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables theta
        ########################################################################
        theta = latent_sample("theta", self.theta_prior)
        y_outcomes = []
        xi_designs = []

        # T-steps experiment
        for t in range(self.T):
            ####################################################################
            # Get a design xi; shape is [num-outer-samples x 1 x 1]
            ####################################################################
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )

            ####################################################################
            # Sample y at xi; shape is [num-outer-samples x 1]
            ####################################################################
            if xi.shape[-1] != self._w_dim:
                xi = xi.view((*xi.shape[:-1], 4))

            mean = self.forward_map(xi, theta)
            sd = self.noise_scale
            y = observation_sample(f"y{t + 1}", dist.Normal(mean, sd).to_event(1))
            y_outcomes.append(y)
            xi_designs.append(xi)

        return y_outcomes

    def forward(self, theta=None):
        """Run the policy"""
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta})
        else:
            model = self.model
        designs = []
        observations = []

        with torch.no_grad():
            trace = pyro.poutine.trace(model).get_trace()
            for t in range(self.T):
                xi = trace.nodes[f"xi{t + 1}"]["value"]
                designs.append(xi)

                y = trace.nodes[f"y{t + 1}"]["value"]
                observations.append(y)
        return torch.cat(designs).unsqueeze(1), torch.cat(observations).unsqueeze(1)

    def eval(self, n_trace=3, theta=None, verbose=True):
        """run the policy, print output and return in a pandas df"""
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta})
        else:
            model = self.model

        output = []
        true_thetas = []
        with torch.no_grad():
            for i in range(n_trace):
                print("\nExample run {}".format(i + 1))
                trace = pyro.poutine.trace(model).get_trace()
                true_theta = trace.nodes["theta"]["value"].cpu()
                if verbose:
                    print(f"*True Theta: {true_theta}*")
                run_xis = []
                run_ys = []
                # Print optimal designs, observations for given theta
                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].cpu().reshape(-1)
                    run_xis.append(xi)
                    y = trace.nodes[f"y{t + 1}"]["value"].cpu().item()
                    run_ys.append(y)
                    if verbose:
                        print(f"xi{t + 1}: {xi}")
                        print(f" y{t + 1}: {y}")

                run_df = pd.DataFrame(torch.stack(run_xis).numpy())
                run_df.columns = [f"xi_{i}" for i in range(self.p)]
                run_df["observations"] = run_ys
                run_df["order"] = list(range(1, self.T + 1))
                run_df["run_id"] = i + 1
                output.append(run_df)
                true_thetas.append(true_theta.numpy())
        print(pd.concat(output))
        return pd.concat(output), true_thetas

    @staticmethod
    def construct_weights(weights: np.array, detector: int = 0):
        """"""

        w_list = [1.0] * 16

        if weights.shape[0] == 4:
            for i in range(4):
                w_list[(8 * detector + i * 2)] = weights[i]
                w_list[(8 * detector + i * 2 + 1)] = weights[i]
        elif weights.shape[0] == 8:
            for i in range(8):
                w_list[(8 * detector + i)] = weights[i]
        else:
            assert False, f"ERROR: Invalid weight array length {weights.shape[0]}. Needs to be 4 or 8."

        return np.array(w_list)


def single_run(
    seed,
    num_steps,
    num_inner_samples,  # L in denom
    num_outer_samples,  # N to estimate outer E
    lr,  # learning rate of adam optim
    gamma,  # scheduler for adam optim
    weight_dim,  # number of physical dim
    K,  # number of sources
    T,  # number of experiments
    noise_scale,
    base_signal,
    max_signal,
    device,
    hidden_dim,
    encoding_dim,
    adam_betas_wd=[0.9, 0.999, 0],  # these are the defaults
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    *adam_betas, adam_weight_decay = adam_betas_wd

    pos, evs = scan_200117()
    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        label=scan_200117.label,
        detector=0,
    )

    ### Set up model ###
    n = 1  # batch dim
    encoder = EncoderNetwork((n, weight_dim), n, hidden_dim, encoding_dim)
    emitter = EmitterNetwork(encoding_dim, (n, weight_dim))
    # Design net: takes pairs [design, observation] as input
    design_net = SetEquivariantDesignNetwork(
        encoder, emitter, empty_value=torch.ones(n, weight_dim) * 0.01
    ).to(device)

    # ----------------------------------------------------------------------------------

    ### Prior hyperparams ###
    # The prior is K independent * p-variate Normals. For example, if there's 1 source
    # (K=1) in 2D (p=2), then we have 1 bivariate Normal.
    theta_prior_loc = torch.zeros((K, weight_dim), device=device)  # mean of the prior
    theta_prior_covmat = torch.eye(weight_dim, device=device)  # covariance of the prior
    # noise of the model: the sigma in N(G(theta, xi), sigma)
    noise_scale_tensor = noise_scale * torch.tensor(
        1.0, dtype=torch.float32, device=device
    )
    # fix the base and the max signal in the G-map
    ho_model = ScanOptDAD(
        scan_map_class=smc,
        design_net=design_net,
        base_signal=base_signal,
        max_signal=max_signal,
        theta_loc=theta_prior_loc,
        theta_covmat=theta_prior_covmat,
        noise_scale=noise_scale_tensor,
        weight_dim=weight_dim,
        K=K,
        T=T,
    )

    ### Set-up optimiser ###
    optimizer = torch.optim.Adam
    # Annealed LR. Set gamma=1 if no annealing required
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {
                "lr": lr,
                "betas": adam_betas,
                "weight_decay": adam_weight_decay,
            },
            "gamma": gamma,
        }
    )
    ### Set-up loss ###
    pce_loss = PriorContrastiveEstimation(num_outer_samples, num_inner_samples)
    oed = OED(ho_model.model, scheduler, pce_loss)

    ### Optimise ###
    loss_history = []
    num_steps_range = trange(0, num_steps, desc="Loss: 0.000 ")
    for i in num_steps_range:
        loss = oed.step()
        loss = torch_item(loss)
        loss_history.append(loss)
        # Log every 50 losses -> too slow (and unnecessary to log everything)
        if i % 50 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))
            # loss_eval = oed.evaluate_loss()
        # Decrease LR at every 1K steps
        if i % 1000 == 0:
            scheduler.step()

    print(loss)
    print(loss_history)
    """
    if len(loss_history) == 0:
        # this happens when we have random designs - there are no grad updates
        loss = torch_item(pce_loss.differentiable_loss(ho_model.model))
    else:
        loss_diff50 = np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1
        loss_av50 = np.mean(loss_history[-51:-1])
    """
    # ho_model.eval()

    return ho_model


ho_model = single_run(
    seed=1234,
    num_steps=20,
    num_inner_samples=1,
    num_outer_samples=1,
    lr=5e-5,
    gamma=0.98,
    device="cpu",
    weight_dim=4,
    K=2,
    T=20,
    noise_scale=0.5,
    base_signal=0.1,
    max_signal=1e-4,
    hidden_dim=256,
    encoding_dim=16,
)

# print(ho_model.forward())
# print(ho_model.eval())
torch.save(ho_model, "ho_model_test")

