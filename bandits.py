import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from xgboost.sklearn import XGBClassifier


def finite_policy_evaluator(A, S, total=None):
    """
    inputs: bandit algorithm A; stream of events S of length L

    S = \{(x, a, r), \dots\}
    x = visit context
    a = action the user actually took
    r = reward (did the action result in a click)
    """

    h = []
    G = 0
    T = 0

    with tqdm(total=total) as pbar:
        for i, (x, a, r) in enumerate(S):
            pbar.update()
            if A(h, x) == a:
                h.append([x, a, r])
                G += r
                T += 1
            if i % 1000 == 0:
                pbar.set_description("Score %0.5f" % ((G / T) if T > 0 else 0))

    return G / T

def parsed_webclick_stream():
    with open('/mnt/md0/data/webscope_user_click_log/parsed.jl') as f:
        for line in f:
            visit = json.loads(line)
            x = visit
            a = visit['displayed_article']
            r = visit['user_clicked']

            yield (x, a, r)

def beta_posterior_lower_bounds(n, s, alpha=1, beta=1):
    """
    A lower bound approximation of the posterior distribution, such
    that for our result x and unknown true parameter theta we have
    P(theta < x) = 0.05, i.e. it's 95% probable that the true theta
    is above our estimate.
     Assuming a Beta(alpha, beta) prior and a Binomial likelihood, we have
    a Beta(alpha + s, beta + n - s) posterior distribution. Rather
    than sampling or inverting the Beta CDF (hard) to find the lower bound
    we use a normal approximation taking
    mu = alpha / (alpha + beta)
    and
    sigma^2 = alpha*beta / (alpha + beta)^2*(alpha + beta + 1),
    we solve 0.05 = \Phi((x-mu) / sigma) for x
    """
    a = alpha + s
    b = beta + n - s
    z = -1.65 # normal z-score lookup given p = 0.05
    return (a / (a + b)) + z * np.sqrt((a*b) / ((a+b)**2 * (a + b + 1)))


def posterior_estimates(samples, successes, choices):
    # TODO these can be done in parallel if needed
    # also could make beta_posterior_lower_bonds operate on the np array of samples & successes
    return [
        beta_posterior_lower_bounds(samples[choice], successes[choice])
        for choice in choices
    ]

class Policy:
    def __init__(self):
        self.samples = defaultdict(int)
        self.successes = defaultdict(int)

    def decide(self, history, visit):
        raise NotImplemented()

    def __call__(self, history, visit):
        """
        history: $\{(x_0, a_0, r_0), \dots, (x_{t-1}, a_{t-1}, r_{t-1})\}$ is the list of past visits which this algorithm has seen.
        visit: is the context of this visit, e.g. session information, user information, etc.
        """
        if len(history) > 1:
            x, a, r = history[-1]
            self.samples[a] += 1
            self.successes[a] += r

        return self.decide(history, visit)

class EpsilonGreedy(Policy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        super().__init__()

    def decide(self, history, visit):
        choices = list(visit['articles'].keys())
        if np.random.uniform(0, 1) < self.epsilon:
            idx = np.random.choice(range(len(choices))) # explore
        else:
            p = posterior_estimates(self.samples, self.successes, choices)
            idx = np.argmax(p) # exploit
        return choices[idx]

class ThompsonSampling(Policy):
    """
    Chooses each arm in proportion to how successful it is.
    """

    def decide(self, history, visit):
        choices = list(visit['articles'].keys())
        p = posterior_estimates(self.samples, self.successes, choices)
        z = (p + np.abs(np.min(p))) # the beta posterior estimate is imperfect and can generate negative numbers
        u = np.ones(len(choices)) / len(choices) # ensure initial choices are uniform, and that future choices never exclude an arm
        z += u
        z /= z.sum() # normalize
        idx = np.random.choice(range(len(choices)), p=z)
        return choices[idx]

class OnlineClassifierPolicy:
    """
    A policy that's tightly coupled to a some classifier
    for it's reward. Always chooses the best reward.

    This is effectively only working in exploitation
    mode, and doesn't have any attempt to explore less
    proven arms, which may not be ideal.
    """

    def __init__(self, clf, burnin = 10):
        self.burnin = burnin
        self.clf = clf

    def context(self, user, visits):
        return np.array([
            user[1:] + v[1:] # clf include intercept
            for v in visits
        ])

    def __call__(self, history, visit):
        if len(history) > 0:
            # TODO might want to setup a training interval so it's not done every single step
            past, a, r = history[-1]
            x = self.context(past['user'], [past['articles'][a]])
            self.clf.partial_fit(x, [r], classes=[0, 1])

        choices = list(visit['articles'].keys())
        if len(history) < self.burnin:
            idx = np.random.choice(range(len(choices)))
        else:
            x = self.context(visit['user'], visit['articles'].values())
            r = self.clf.predict(x)
            idx = np.argmax(r)

        return choices[idx]

if __name__ == "__main__":
    N = 45811883

    print('EpsilonGreedy')
    finite_policy_evaluator(EpsilonGreedy(), parsed_webclick_stream(), total=N)

    print('ThompsonSampling')
    finite_policy_evaluator(ThompsonSampling(), parsed_webclick_stream(), total=N)

    print('SGD')
    clf = SGDClassifier()
    sgd = OnlineClassifierPolicy(clf)
    finite_policy_evaluator(sgd, parsed_webclick_stream(), total=N)

    print('XGB')
    xgboost_clf = XGBClassifier()
    xgb_policy = OnlineClassifierPolicy(clf)
    finite_policy_evaluator(xgb_policy, parsed_webclick_stream(), total=N)
