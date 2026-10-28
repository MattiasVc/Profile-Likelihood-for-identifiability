import numpy as np
from scipy.integrate import odeint
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.stats import qmc
import matplotlib.pyplot as plt
from numpy import random as rdm
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool 
from os import cpu_count

class Identifiability_Analysis_lotka_volterra:
    # Parameters (log10 scale): alpha, beta, delta, gamma
    # States: y = [prey, predator]; initial populations are known.

    PARAM_NAMES = ["alpha", "beta", "delta", "gamma"]

    def __init__(
        self,
        n_data_points,
        opt_pm=None,
        sigma=0.1,
        n_pm_search_lattice_points=15,
        observed=(0, 1),
        true_params = (0.52, 0.026, 0.026, 0.84),
        t_end = 20,
        bool_unknown_y0 = False,
        confidence_treshold = 3.87 # 95 percent confidence intervals
    ):
        if (
            not isinstance(n_data_points, (int, np.integer))
            or isinstance(n_data_points, bool)
            or n_data_points < 1
        ):
            raise ValueError("n_data_points must be a positive integer")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if not observed or any(index not in (0, 1) for index in observed):
            raise ValueError("observed must contain 0 (prey), 1 (predator), or both")
        if len(set(observed)) != len(observed):
            raise ValueError("observed indices must not be repeated")
        if n_pm_search_lattice_points < 1:
            raise ValueError("n_pm_search_lattice_points must be positive")

        self.n_data_points = n_data_points
        self.sigma_fraction = sigma
        self.n_pm_search_lattice_points = n_pm_search_lattice_points
        self.observed = list(observed)
        self.t_end = t_end

        # Optional: treat the initial populations as unknown (last two parameters)
        self.bool_unknown_y0 = bool_unknown_y0
        self.n_params = 6 if bool_unknown_y0 else 4
        self.PARAM_NAMES = list(type(self).PARAM_NAMES)
        self.y0 = np.array([34.0, 5.9])  # true initial prey and predator populations
        self.true_params = tuple(true_params)  # alpha, beta, delta, gamma
        if bool_unknown_y0:
            self.PARAM_NAMES += ["prey_0", "predator_0"]
            self.true_params += tuple(self.y0)
        self.labels = ["Prey", "Predator"]
        self.colors = ["darkorange", "purple"]
        self.PHI_BOUND_LOWER = -2
        self.PHI_BOUND_UPPER = 0.5
        # Per-parameter log10 bounds; initial populations are searched in [1, 100]
        self.phi_lower = np.full(self.n_params, self.PHI_BOUND_LOWER, dtype=float)
        self.phi_upper = np.full(self.n_params, self.PHI_BOUND_UPPER, dtype=float)
        self.phi_lower[4:] = 0.0
        self.phi_upper[4:] = 2.0
        self.threshold_value = confidence_treshold

        self.t_int, self.t_exp, self.states, self.exp_data, self.sigma = (
            self.generate_synthetic_data()
        )

        self.phi_opt = None
        self.cost_opt = None
        if opt_pm is not None:
            self.phi_opt = np.asarray(opt_pm, dtype=float)
            if self.phi_opt.shape != (self.n_params,):
                raise ValueError(f"opt_pm must contain {self.n_params} log10 parameters")
            self.cost_opt = self.optimization_function(self.phi_opt)

        self.ppl_data = {}

    def model(self, y, t, alpha, beta, delta, gamma):
        prey, predator = y
        dprey = alpha * prey - beta * prey * predator
        dpredator = delta * prey * predator - gamma * predator
        return [dprey, dpredator]

    def generate_synthetic_data(self):
        """Generate noisy prey and predator observations at evenly spaced times."""
        t_int = np.linspace(0, self.t_end, 10 * self.t_end + 1)
        t_exp = np.linspace(0, self.t_end, self.n_data_points)
        states = odeint(self.model, self.y0, t_int, args=self.true_params[:4])
        states_at_obs = odeint(self.model, self.y0, t_exp, args=self.true_params[:4])

        observed_data = states_at_obs[:, self.observed]
        noise_sigma = self.sigma_fraction * np.max(observed_data, axis=0)
        rng = rdm.RandomState(4)
        exp_data = observed_data + rng.normal(
            loc=0,
            scale=noise_sigma,
            size=observed_data.shape,
        )
        arr_sigma = np.broadcast_to(noise_sigma, exp_data.shape).copy()
        """
        fig, ax = plt.subplots()
        for data_index, state_index in enumerate(self.observed):
            ax.scatter(
                t_exp,
                exp_data[:, data_index],
                label=f"{self.labels[state_index]} experimental data",
                s=5,
                color=self.colors[state_index],
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.legend()
        plt.show()
        """
        return t_int, t_exp, states, exp_data, arr_sigma
    
    def plot_true_trajectory(self):
        fig, ax = plt.subplots()
        for state_index in range(2):
            ax.plot(
                self.t_int,
                self.states[:, state_index],
                label=f"{self.labels[state_index]} true trajectory",
                color=self.colors[state_index],
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.legend()
        plt.show()

    def init_phi(self, n_lattice_points=None):
        """Initialize log10-parameter guesses for alpha, beta, delta and gamma."""
        if n_lattice_points is None:
            n_lattice_points = self.n_pm_search_lattice_points
        sampler = qmc.LatinHypercube(d=self.n_params)
        sample = sampler.random(n_lattice_points)
        spacing = self.phi_upper - self.phi_lower
        return [
            sample[i, :] * spacing + self.phi_lower
            for i in range(sample.shape[0])
        ]

    def solve_single(self, phi):
        """Solve the Lotka–Volterra system for one log10 parameter vector."""
        theta = 10 ** np.asarray(phi, dtype=float)
        y0 = theta[4:6] if self.bool_unknown_y0 else self.y0
        states_fit = odeint(self.model, y0, self.t_int, args=tuple(theta[:4]))
        states_at_obs = odeint(self.model, y0, self.t_exp, args=tuple(theta[:4]))
        return states_at_obs, states_fit

    def optimization_function(self, phi):
        """Evaluate the Gaussian -2 log-likelihood for the synthetic data."""
        states_at_obs, _ = self.solve_single(phi)
        residuals = self.exp_data - states_at_obs[:, self.observed]
        sigma_safe = np.maximum(self.sigma, 1e-9)
        minus_2_ll = np.sum(
            np.log(2 * np.pi * sigma_safe**2) + residuals**2 / sigma_safe**2
        )
        return float(minus_2_ll) if np.isfinite(minus_2_ll) else 1e12

    def calibrate_trust_region(self):
        """Multi-start trust-region calibration for all four model parameters."""
        phi_bounds = Bounds(self.phi_lower, self.phi_upper)
        results = []
        print("iteration:")
        for iteration, phi0 in enumerate(self.init_phi()):
            print(iteration)
            results.append(
                minimize(
                    self.optimization_function,
                    phi0,
                    method="trust-constr",
                    bounds=phi_bounds,
                )
            )

        costs = np.array([result.fun for result in results])
        best_idx = np.argmin(costs)
        phi_opt = results[best_idx].x
        cost_opt = costs[best_idx]

        sorted_costs = np.sort(costs)
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(costs)), sorted_costs, lw=1, color="black")
        ax.scatter(np.arange(len(costs)), sorted_costs, s=5, color="red")
        ax.set_xlabel("index")
        ax.set_ylabel("Objective function")
        if np.all(sorted_costs > 0):
            ax.set_yscale("log")
        plt.show()

        print("TR phi_opt:", phi_opt)
        print("TR theta_opt:", 10**phi_opt, " true:", self.true_params)
        print("TR cost_opt:", cost_opt)
        self.phi_opt = np.copy(phi_opt)
        self.cost_opt = float(cost_opt)
        output_dir = self.get_output_dir()
        np.savetxt(output_dir / "phi_opt.txt", phi_opt)
        np.savetxt(output_dir / "cost_opt.txt", [cost_opt])
        return phi_opt, cost_opt

    def plot_result(self, phi_opt=None):
        """Plot fitted population trajectories and the observed data."""
        if phi_opt is None:
            phi_opt = self.phi_opt = np.loadtxt(self.get_output_dir() / "phi_opt.txt")
            self.cost_opt = float(np.loadtxt(self.get_output_dir() / "cost_opt.txt"))
        _, states_fit = self.solve_single(phi_opt)
        fig, ax = plt.subplots()
        for state_index in range(2):
            ax.plot(
                self.t_int,
                states_fit[:, state_index],
                color=self.colors[state_index],
                lw=1,
                label=self.labels[state_index] + " fit",
            )
        for data_index, state_index in enumerate(self.observed):
            ax.errorbar(
                self.t_exp,
                self.exp_data[:, data_index],
                yerr=self.sigma[:, data_index],
                fmt="none",
                color="grey",
                alpha=0.4,
            )
            ax.scatter(
                self.t_exp,
                self.exp_data[:, data_index],
                color=self.colors[state_index],
                s=10,
                label=self.labels[state_index] + " data",
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.legend()
        plt.savefig("trajectory_fit")
        plt.show()

    def recalibrate_model_simple(self, fixed_val, fixed_index, phi_opt):
        """Re-optimize all other parameters with one parameter fixed."""
        if not 0 <= fixed_index < self.n_params:
            raise ValueError(f"fixed_index must be between 0 and {self.n_params - 1}")
        if fixed_val <= 0:
            raise ValueError("fixed_val must be positive")

        phi_opt = np.asarray(phi_opt, dtype=float)
        phi0 = np.delete(phi_opt, fixed_index)
        fixed_phi = np.log10(fixed_val)
        if not self.phi_lower[fixed_index] <= fixed_phi <= self.phi_upper[fixed_index]:
            raise ValueError("fixed_val must lie within the PHI bounds")

        def objective(phi_free):
            return self.optimization_function(
                np.insert(phi_free, fixed_index, fixed_phi)
            )

        bounds = Bounds(
            np.delete(self.phi_lower, fixed_index),
            np.delete(self.phi_upper, fixed_index),
        )
        result = minimize(objective, phi0, method="trust-constr", bounds=bounds)
        phi_opt_new = np.insert(result.x, fixed_index, fixed_phi)
        return phi_opt_new, result.fun

    def profile_likelihood_simple_(
        self,
        fixed_index,
        n_points=15,
        lower_range=1,
        upper_range=1,
        bool_all_range=False,
        show_plot=False,
    ):
        """Profile one parameter at logarithmically spaced values."""
        if self.phi_opt is None:
            raise ValueError(
                "The parameters have to be estimated first (calibrate_trust_region)"
            )
        if not 0 <= fixed_index < self.n_params:
            raise ValueError(f"fixed_index must be between 0 and {self.n_params - 1}")

        phi_opt = np.asarray(self.phi_opt, dtype=float)
        param_opt = phi_opt[fixed_index]
        if bool_all_range:
            lower_bound = self.phi_lower[fixed_index]
            upper_bound = self.phi_upper[fixed_index]
        else:
            lower_bound = max(param_opt - lower_range, self.phi_lower[fixed_index])
            upper_bound = min(param_opt + upper_range, self.phi_upper[fixed_index])
        if lower_bound >= upper_bound:
            raise ValueError("The requested profile interval does not overlap the PHI bounds")

        fixed_param_values = 10 ** np.linspace(lower_bound, upper_bound, n_points)
        chi_sq_values = []
        print(f"\nComputing profile likelihood for {self.PARAM_NAMES[fixed_index]}...")
        print(f"Bounds (log10): [{lower_bound:.3f}, {upper_bound:.3f}]")

        phi_init = phi_opt
        for fixed_val in fixed_param_values:
            phi_opt_new, chi_sq_val = self.recalibrate_model_simple(
                fixed_val, fixed_index, phi_init
            )
            chi_sq_values.append(chi_sq_val)
            phi_init = phi_opt_new

        ppl_data = np.column_stack((fixed_param_values, chi_sq_values))
        self.ppl_data[fixed_index] = ppl_data.copy()
        output_dir = self.get_output_dir()
        np.savetxt(
            output_dir / f"ppl_data_{self.PARAM_NAMES[fixed_index]}_N_{n_points}_points.txt",
            ppl_data,
        )
        if show_plot:
            self.plot_ppl_data(ppl_data, fixed_index, n_points)
        return ppl_data

    def get_output_dir(self):
        """Single output folder for profile data and figures of this configuration."""
        add_str = "both" if len(self.observed) == 2 else str(self.observed[0])
        if self.bool_unknown_y0:
            add_str += "_y0_unknown"
        output_dir = Path("FiguresProfileLikelihood_lotka_volterra_" + add_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def plot_ppl_data(self, ppl_data, fixed_index, n_points, ax=None):
        """Plot profile likelihood with fitted and true parameter values.

        If `ax` is given, draw on it only (no saving/showing); otherwise create,
        save and show a standalone figure.
        """
        if not 0 <= fixed_index < self.n_params:
            raise ValueError(f"fixed_index must be between 0 and {self.n_params - 1}")

        standalone = ax is None
        fixed_param_values = ppl_data[:, 0]
        chi_sq_values = ppl_data[:, 1]
        if standalone:
            fig, ax = plt.subplots()
        ax.axhline(
            y=self.cost_opt + self.threshold_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"cost_opt + {self.threshold_value}",
        )
        ax.axvline(
            x=np.asarray(self.phi_opt, dtype=float)[fixed_index],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="Optimal value",
        )
        ax.axvline(
            x=np.log10(self.true_params[fixed_index]),
            color="blue",
            linestyle=":",
            linewidth=1.5,
            label="True value",
        )
        ax.plot(np.log10(fixed_param_values), chi_sq_values, color="red", linewidth=1.5)
        ax.set_xlabel(f"log10({self.PARAM_NAMES[fixed_index]})", fontsize = 18)
        ax.set_ylabel("Profile likelihood value")
        ax.set_ylim(self.cost_opt - 3, self.cost_opt + self.threshold_value + 10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if not standalone:
            return
        output_dir = self.get_output_dir()
        plt.savefig(output_dir / f"_n_points_{n_points}_"
            f"fidx_{fixed_index}_sigma_{self.sigma_fraction}_Threshold_{self.threshold_value}.png")
        np.savetxt(
            output_dir / f"ppl_data_{self.PARAM_NAMES[fixed_index]}_N_{n_points}_points.txt",
            ppl_data,
        )
        plt.show()
    
    def extract_confidence_intervals( self, arr_fixed_indices=None, arr_n_points=None ):
        
        if self.cost_opt is None:
            raise ValueError("Calibrate the model before extracting confidence intervals")
        if arr_fixed_indices is None:
            arr_fixed_indices = range(self.n_params)
        if arr_n_points is None:
            arr_n_points = [15] * len(arr_fixed_indices)
        if len(arr_fixed_indices) != len(arr_n_points):
            raise ValueError("arr_fixed_indices and arr_n_points must have the same length")

        output_dir = self.get_output_dir()

        critical_value = self.cost_opt + self.threshold_value
        print(f"Confidence intervals (cost threshold = {critical_value:.4g}), "
            f"data from '{output_dir}'")
        confidence_intervals = {}

        n_plots = len(arr_fixed_indices)
        n_cols = 2
        n_rows = max(1, int(np.ceil(n_plots / n_cols)))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
        axs = axs.flatten()
        for ax in axs[n_plots:]:
            ax.set_visible(False)

        for ax, fixed_index, n_points in zip(axs, arr_fixed_indices, arr_n_points):
            name = self.PARAM_NAMES[fixed_index]
            path = output_dir / f"ppl_data_{name}_N_{n_points}_points.txt"
            if not path.exists():
                print(f"{name:>6}: file not found ({path.name})")
                ax.set_visible(False)
                continue

            profile = np.loadtxt(path, ndmin=2)
            values = profile[profile[:, 1] <= critical_value, 0]

            self.plot_ppl_data(profile, fixed_index, n_points, ax=ax)

            if values.size == 0:
                print(f"{name:>6}: no profile points within the confidence threshold")
                continue

            low, high = float(values.min()), float(values.max())
            confidence_intervals[name] = (low, high)
            print(
                f"{name:>6}: [{low:.3e}, {high:.3e}]  "
                f"log10: [{np.log10(low):+.2f}, {np.log10(high):+.2f}]  "
                f"span: {np.log10(high) - np.log10(low):.2f} orders of magnitude"
            )

        fig.tight_layout()
        
        output_dir = self.get_output_dir()
        plt.savefig(output_dir / f"All_Profiles")
        plt.show()
        return confidence_intervals

if __name__ == "__main__":

    # Parameter estimation:

    # CASE 1
    ID_LV = Identifiability_Analysis_lotka_volterra(14, sigma = 0.01, n_pm_search_lattice_points=50)
    ID_LV.calibrate_trust_region()
    ID_LV.plot_result()
    #----------------------
   
    # CASE 2
    #14 data_points:
    ID_LV = Identifiability_Analysis_lotka_volterra(14, sigma = 0.01, n_pm_search_lattice_points=50, observed = (0,) )
    ID_LV.calibrate_trust_region()
    ID_LV.plot_result()
    #------------------------------------------------
    
    ID_LV = Identifiability_Analysis_lotka_volterra(14, sigma = 0.01, n_pm_search_lattice_points=50, observed = (1,) )
    ID_LV.calibrate_trust_region()
    ID_LV.plot_result()


    # CASE 3
    ID_LV = Identifiability_Analysis_lotka_volterra(14, sigma = 0.01, n_pm_search_lattice_points=50, bool_unknown_y0=True, observed=(0,) )
    ID_LV.calibrate_trust_region()
    ID_LV.plot_result()


