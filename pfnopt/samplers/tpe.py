import numpy
import scipy.special
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Callable  # NOQA
from typing import Tuple  # NOQA
from typing import Union  # NOQA

from pfnopt import distributions
from pfnopt.samplers import base
from pfnopt.samplers import random
from pfnopt.storages.base import BaseStorage  # NOQA


default_consider_prior = True
default_prior_weight = 1.0
default_consider_magic_clip = True
default_consider_endpoints = False
default_n_startup_trials = 4
default_n_ei_candidates = 24
EPS = 1e-12


def default_gamma(x):
    return min(int(numpy.ceil(0.25 * numpy.sqrt(x))), 25)


def default_weights(x):
    if x == 0:
        return numpy.asarray([])
    elif x < 25:
        return numpy.ones(x)
    else:
        ramp = numpy.linspace(1.0 / x, 1.0, num=x-25)
        flat = numpy.ones(25)
        return numpy.concatenate([ramp, flat], axis=0)


class TPESampler(base.BaseSampler):

    def __init__(self,
                 consider_prior=default_consider_prior,
                 prior_weight=default_prior_weight,
                 consider_magic_clip=default_consider_magic_clip,
                 consider_endpoints=default_consider_endpoints,
                 n_startup_trials=default_n_startup_trials,
                 n_ei_candidates=default_n_ei_candidates,
                 gamma=default_gamma,
                 weights=default_weights,
                 seed=None):
        # type: (bool, float, bool, bool, int, int, Callable, Callable, Optional[int]) -> None

        self.consider_prior = consider_prior
        self.prior_weight = prior_weight
        self.consider_magic_clip = consider_magic_clip
        self.consider_endpoints = consider_endpoints
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.weights = weights
        self.seed = seed

        self.rng = numpy.random.RandomState(seed)
        self.random_sampler = random.RandomSampler(seed=seed)

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, distributions.BaseDistribution) -> float

        observation_pairs = storage.get_trial_param_result_pairs(
            study_id, param_name)
        n = len(observation_pairs)

        # TODO(Akiba): this behavior is slightly different from hyperopt
        if n < self.n_startup_trials:
            return self.random_sampler.sample(storage, study_id, param_name, param_distribution)

        below_param_values, above_param_values = self._split_observation_pairs(
            list(range(n)), [p[0] for p in observation_pairs],
            list(range(n)), [p[1] for p in observation_pairs])

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self._sample_uniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            return self._sample_loguniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            return self._sample_discrete_uniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            return self._sample_int(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            return self._sample_categorical(
                param_distribution, below_param_values, above_param_values)
        else:
            raise NotImplementedError

    def _split_observation_pairs(self, config_idxs, config_vals, loss_idxs, loss_vals):
        # type: (List[int], List[float], List[int], List[float]) -> (List[float], List[float])

        config_idxs, config_vals, loss_idxs, loss_vals = map(numpy.asarray,
                                                             [config_idxs, config_vals, loss_idxs, loss_vals])
        assert len(config_idxs) == len(config_vals) == len(loss_idxs) == len(loss_vals)
        n_below = self.gamma(len(config_vals))
        loss_ascending = numpy.argsort(loss_vals)

        keep_idxs = set(loss_idxs[loss_ascending[:n_below]])
        below = [v for i, v in zip(config_idxs, config_vals) if i in keep_idxs]

        keep_idxs = set(loss_idxs[loss_ascending[n_below:]])
        above = [v for i, v in zip(config_idxs, config_vals) if i in keep_idxs]

        return below, above

    def _sample_uniform(self, distribution, below, above):
        # type: (distributions.UniformDistribution, List[float], List[float]) -> float

        size = (self.n_ei_candidates,)
        weights_b, mus_b, sigmas_b = self._make_parzen_estimator(mus=below,
                                                                 low=distribution.low,
                                                                 high=distribution.high)

        samples_b = self._GMM(weights=weights_b,
                              mus=mus_b,
                              sigmas=sigmas_b,
                              low=distribution.low,
                              high=distribution.high,
                              q=None,
                              size=size)

        log_likelihoods_b = self._GMM_log_pdf(samples=samples_b,
                                              weights=weights_b,
                                              mus=mus_b,
                                              sigmas=sigmas_b,
                                              low=distribution.low,
                                              high=distribution.high,
                                              q=None)

        weights_a, mus_a, sigmas_a = self._make_parzen_estimator(mus=above,
                                                                 low=distribution.low,
                                                                 high=distribution.high)

        log_likelihoods_a = self._GMM_log_pdf(samples=samples_b,
                                              weights=weights_a,
                                              mus=mus_a,
                                              sigmas=sigmas_a,
                                              low=distribution.low,
                                              high=distribution.high,
                                              q=None)

        return self._compare(samples=samples_b, log_l=log_likelihoods_b, log_g=log_likelihoods_a)[0]

    def _sample_loguniform(self, distribution, below, above):
        # type: (distributions.LogUniformDistribution, List[float], List[float]) -> float

        low = numpy.log(distribution.low)
        high = numpy.log(distribution.high)
        size = (self.n_ei_candidates,)
        # From below, make samples and log-likelihoods
        weights_b, mus_b, sigmas_b = self._make_parzen_estimator(mus=numpy.log(below),
                                                                 low=low,
                                                                 high=high)
        samples_b = self._GMM(weights=weights_b,
                              mus=mus_b,
                              sigmas=sigmas_b,
                              low=low,
                              high=high,
                              q=None,
                              is_log=True,
                              size=size)
        log_likelihoods_b = self._GMM_log_pdf(samples=samples_b,
                                              weights=weights_b,
                                              mus=mus_b,
                                              sigmas=sigmas_b,
                                              low=low,
                                              high=high,
                                              q=None,
                                              is_log=True)

        # From above, make log-likelihoods
        weights_a, mus_a, sigmas_a = self._make_parzen_estimator(mus=numpy.log(above),
                                                                 low=low,
                                                                 high=high)

        log_likelihoods_a = self._GMM_log_pdf(samples=samples_b,
                                              weights=weights_a,
                                              mus=mus_a,
                                              sigmas=sigmas_a,
                                              low=low,
                                              high=high,
                                              q=None,
                                              is_log=True)

        return self._compare(samples=samples_b, log_l=log_likelihoods_b, log_g=log_likelihoods_a)[0]

    def _sample_discrete_uniform(self, distribution, below, above):
        # type: (distributions.DiscreteUniformDistribution, List[float], List[float]) -> float

        low = distribution.low - 0.5 * distribution.q
        high = distribution.high + 0.5 * distribution.q
        size = (self.n_ei_candidates,)

        weights_b, mus_b, sigmas_b = self._make_parzen_estimator(mus=below,
                                                                 low=low,
                                                                 high=high)
        samples_b = self._GMM(weights=weights_b,
                              mus=mus_b,
                              sigmas=sigmas_b,
                              low=low,
                              high=high,
                              q=distribution.q,
                              size=size)
        log_likelihoods_b = self._GMM_log_pdf(samples=samples_b,
                                              weights=weights_b,
                                              mus=mus_b,
                                              sigmas=sigmas_b,
                                              low=low,
                                              high=high,
                                              q=distribution.q)

        weights_a, mus_a, sigmas_a = self._make_parzen_estimator(mus=above,
                                                                 low=low,
                                                                 high=high)

        log_likelihoods_a = self._GMM_log_pdf(samples=samples_b,
                                              weights=weights_a,
                                              mus=mus_a,
                                              sigmas=sigmas_a,
                                              low=low,
                                              high=high,
                                              q=distribution.q)

        return min(max(self._compare(samples=samples_b, log_l=log_likelihoods_b, log_g=log_likelihoods_a)[0], low), high)

    def _sample_int(self, distribution, below, above):
        # type: (distributions.IntUniformDistribution, List[float], List[float]) -> float

        distribution = distributions.DiscreteUniformDistribution(low=distribution.low, high=distribution.high, q=1.0)
        v = self._sample_discrete_uniform(distribution, below, above)
        return int(v)

    def _sample_categorical(self, distribution, below, above):
        # type: (distributions.CategoricalDistribution, List[float], List[float]) -> float
        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        upper = len(choices)
        size = (self.n_ei_candidates,)

        weights_b = self.weights(len(below))
        counts_b = numpy.bincount(below, minlength=upper, weights=weights_b)
        pseudocounts_b = counts_b + self.prior_weight
        pseudocounts_b /= pseudocounts_b.sum()
        samples_b = self._categorical(pseudocounts_b, size=size)
        log_likelihoods_b = self._categorical_log_pdf(samples_b, pseudocounts_b)

        weights_a = self.weights(len(above))
        counts_a = numpy.bincount(above, minlength=upper, weights=weights_a)
        pseudocounts_a = counts_a + self.prior_weight
        pseudocounts_a /= pseudocounts_a.sum()
        log_likelihoods_a = self._categorical_log_pdf(samples_b, pseudocounts_a)

        return int(self._compare(samples=samples_b, log_l=log_likelihoods_b, log_g=log_likelihoods_a)[0])

    def _make_parzen_estimator(self, mus, low, high):
        # type: (List[float], float, float) -> (List[float], List[float], List[float])

        mus = numpy.asarray(mus)

        if self.consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low)
            if len(mus) == 0:
                sorted_mus = numpy.asarray([prior_mu])
                sigma = numpy.asarray([prior_sigma])
                prior_pos = 0
                order = []
            elif len(mus) == 1:
                if prior_mu < mus[0]:
                    prior_pos = 0
                    sorted_mus = numpy.asarray([prior_mu, mus[0]])
                    sigma = numpy.asarray([prior_sigma, prior_sigma * .5])
                else:
                    prior_pos = 1
                    sorted_mus = numpy.asarray([mus[0], prior_mu])
                    sigma = numpy.asarray([prior_sigma * .5, prior_sigma])
                order = [0]
            else:  # len(mus) >= 2
                # decide where prior is placed
                order = numpy.argsort(mus)
                prior_pos = numpy.searchsorted(mus[order], prior_mu)

                # decide mus
                sorted_mus = numpy.zeros(len(mus) + 1)
                sorted_mus[:prior_pos] = mus[order[:prior_pos]]
                sorted_mus[prior_pos] = prior_mu
                sorted_mus[prior_pos + 1:] = mus[order[prior_pos:]]

                # decide sigmas
                low_sorted_mus_high = numpy.append(sorted_mus, high)
                low_sorted_mus_high = numpy.insert(low_sorted_mus_high, 0, low)
                sigma = numpy.zeros_like(low_sorted_mus_high)
                sigma[1:-1] = numpy.maximum(low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                                            low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1])
                if not self.consider_endpoints:
                    sigma[1] = sigma[2] - sigma[1]
                    sigma[-2] = sigma[-2] - sigma[-3]
                sigma = sigma[1:-1]

            # decide weights
            unsorted_weights = self.weights(len(mus))
            sorted_weights = numpy.zeros_like(sorted_mus)
            sorted_weights[:prior_pos] = unsorted_weights[order[:prior_pos]]
            sorted_weights[prior_pos] = self.prior_weight
            sorted_weights[prior_pos + 1:] = unsorted_weights[order[prior_pos:]]
            sorted_weights /= sorted_weights.sum()
        else:
            order = numpy.argsort(mus)

            # decide mus
            sorted_mus = mus[order]

            # decide sigmas
            if len(mus) == 0:
                sigma = []
            else:
                low_sorted_mus_high = numpy.append(sorted_mus, high)
                low_sorted_mus_high = numpy.insert(low_sorted_mus_high, 0, low)
                sigma = numpy.zeros_like(low_sorted_mus_high)
                sigma[1:-1] = numpy.maximum(low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                                            low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1])
                if not self.consider_endpoints:
                    sigma[1] = sigma[2] - sigma[1]
                    sigma[-2] = sigma[-2] - sigma[-3]
                sigma = sigma[1:-1]

            # decide weights
            unsorted_weights = self.weights(len(mus))
            sorted_weights = unsorted_weights[order]
            sorted_weights /= sorted_weights.sum()

        if self.consider_magic_clip:
            maxsigma = 1.0 * (high - low)
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(sorted_mus)))
            sigma = numpy.clip(sigma, minsigma, maxsigma)
        else:
            maxsigma = 1.0 * (high - low)
            minsigma = 0.0
            sigma = numpy.clip(sigma, maxsigma, minsigma)

        return sorted_weights, sorted_mus, sigma

    def _GMM(self, weights, mus, sigmas, low, high, q=None, size=(), is_log=False):
        # type: (List[float], List[float], List[float], float, float, Optional[float], Tuple, bool) -> numpy.ndarray

        weights, mus, sigmas = map(numpy.asarray, (weights, mus, sigmas))
        n_samples = numpy.prod(size)

        if low >= high:
            raise ValueError("low >= high", (low, high))
        samples = []
        while len(samples) < n_samples:
            active = numpy.argmax(self.rng.multinomial(1, weights))
            draw = self.rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw <= high:
                samples.append(draw)
        samples = numpy.reshape(numpy.asarray(samples), size)

        if is_log:
            samples = numpy.exp(samples)

        if q is None:
            return samples
        else:
            return numpy.round(samples / q) * q

    def _GMM_log_pdf(self, samples, weights, mus, sigmas, low, high, q=None, is_log=False):
        # type: (List[float], List[float], List[float], List[float], float, float, Optional[float], bool) -> List[float]

        samples, weights, mus, sigmas = map(numpy.asarray, (samples, weights, mus, sigmas))
        if samples.size == 0:
            return numpy.asarray([])
        if weights.ndim != 1:
            raise TypeError("need vector of weights", weights.shape)
        if mus.ndim != 1:
            raise TypeError("need vector of mus", mus.shape)
        if sigmas.ndim != 1:
            raise TypeError("need vector of sigmas", sigmas.shape)
        _samples = samples
        samples = _samples.flatten()

        p_accept = numpy.sum(weights * (self._normal_cdf(high, mus, sigmas) - self._normal_cdf(low, mus, sigmas)))

        if q is None:
            if is_log:
                distance = numpy.log(samples[:, None]) - mus
                mahalanobis = (distance / numpy.maximum(sigmas, EPS)) ** 2
                Z = numpy.sqrt(2 * numpy.pi) * sigmas * samples[:, None]
                coefficient = weights / Z / p_accept
                return_val = self._logsum_rows(- 0.5 * mahalanobis + numpy.log(coefficient))
            else:
                distance = samples[:, None] - mus
                mahalanobis = (distance / numpy.maximum(sigmas, EPS)) ** 2
                Z = numpy.sqrt(2 * numpy.pi) * sigmas
                coefficient = weights / Z / p_accept
                return_val = self._logsum_rows(- 0.5 * mahalanobis + numpy.log(coefficient))
        else:
            probabilities = numpy.zeros(samples.shape, dtype='float64')
            for w, mu, sigma in zip(weights, mus, sigmas):
                if is_log:
                    upper_bound = numpy.minimum(samples + q / 2.0, numpy.exp(high))
                    lower_bound = numpy.maximum(samples - q / 2.0, numpy.exp(low))
                    lower_bound = numpy.maximum(0, lower_bound)
                    inc_amt = w * self._log_normal_cdf(upper_bound, mu, sigma)
                    inc_amt -= w * self._log_normal_cdf(lower_bound, mu, sigma)
                else:
                    upper_bound = numpy.minimum(samples + q / 2.0, high)
                    lower_bound = numpy.maximum(samples - q / 2.0, low)
                    inc_amt = w * self._normal_cdf(upper_bound, mu, sigma)
                    inc_amt -= w * self._normal_cdf(lower_bound, mu, sigma)
                probabilities += inc_amt
            return_val = numpy.log(probabilities) - numpy.log(p_accept)

        return_val.shape = _samples.shape
        return return_val

    def _categorical(self, p, size=()):
        # type: (Union[List[float], numpy.ndarray], Tuple) -> Union[List[float], numpy.ndarray]

        if len(p) == 1 and isinstance(p[0], numpy.ndarray):
            p = p[0]
        p = numpy.asarray(p)

        if size == ():
            size = (1,)
        elif isinstance(size, (int, numpy.number)):
            size = (size,)
        else:
            size = tuple(size)

        if size == (0,):
            return numpy.asarray([])
        assert len(size)

        if p.ndim == 0:
            raise NotImplementedError
        elif p.ndim == 1:
            n_draws = int(numpy.prod(size))
            sample = self.rng.multinomial(n=1, pvals=p, size=int(n_draws))
            assert sample.shape == size + (len(p),)
            return_val = numpy.dot(sample, numpy.arange(len(p)))
            return_val.shape = size
            return return_val
        elif p.ndim == 2:
            n_draws_, n_choices = p.shape
            n_draws, = size
            assert n_draws_ == n_draws
            return_val = [numpy.where(self.rng.multinomial(pvals=[ii], n=1))[0][0] for ii in range(n_draws_)]
            return_val = numpy.asarray(return_val)
            return_val.shape = size
            return return_val
        else:
            raise NotImplementedError

    def _categorical_log_pdf(self, sample, p):
        # type: (Union[List[float], numpy.ndarray], List[float]) -> Union[List[float], numpy.ndarray]

        if sample.size:
            return numpy.log(numpy.asarray(p)[sample])
        else:
            return numpy.asarray([])

    def _compare(self, samples, log_l, log_g):
        # type: (List[float], List[float], List[float]) -> List[float]

        samples, log_l, log_g = map(numpy.asarray, (samples, log_l, log_g))
        if len(samples):
            score = log_l - log_g
            if len(samples) != len(score):
                raise ValueError()
            best = numpy.argmax(score)
            return [samples[best]] * len(samples)
        else:
            return []

    def _logsum_rows(self, x):
        # type: (List[float]) -> numpy.ndarray

        x = numpy.asarray(x)
        m = x.max(axis=1)
        return numpy.log(numpy.exp(x - m[:, None]).sum(axis=1)) + m

    def _normal_cdf(self, x, mu, sigma):
        # type: (float, List[float], List[float]) -> numpy.ndarray

        mu, sigma = map(numpy.asarray, (mu, sigma))
        top = x - mu
        bottom = numpy.maximum(numpy.sqrt(2) * sigma, EPS)
        z = top / bottom
        return 0.5 * (1 + scipy.special.erf(z))

    def _lognormal_cdf(self, x, mu, sigma):
        if x.min() < 0:
            raise ValueError("negative argument is given to _lognormal_cdf", x)
        olderr = numpy.seterr(divide='ignore')
        try:
            top = numpy.log(numpy.maximum(x, EPS)) - mu
            bottom = numpy.maximum(numpy.sqrt(2) * sigma, EPS)
            z = top / bottom
            return .5 + .5 * scipy.special.erf(z)
        finally:
            numpy.seterr(**olderr)
