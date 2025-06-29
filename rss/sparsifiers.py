import numpy as np

def pivotal(v, m):
    rng = np.random.default_rng()

    nonzeros = v.nonzero()[0]
    N_reduced = nonzeros.size
    result = np.copy(v)

    if m < N_reduced:
        reduced = v[nonzeros]
        mag = np.abs(reduced)

        keep = np.zeros(N_reduced, dtype=bool)
        threshold = 0.0

        # Step 1: Exact preservation
        for _ in range(m):
            remaining = ~keep
            num_remaining = np.sum(remaining)
            if num_remaining == 0:
                break
            threshold = np.sum(mag[remaining]) / num_remaining
            new = (mag > threshold * (1 - 1e-8))
            if np.sum(new) > np.sum(keep):
                keep = new
            else:
                break

        # Step 2: Probabilistic amplification
        if threshold > 0 and np.sum(keep) < m:
            probs = mag / threshold
            probs[keep] = 0

            candidate_inds = np.flatnonzero(probs > 0)
            if candidate_inds.size == 0:
                reduced[~keep] = 0
                result[nonzeros] = reduced
                return result

            amplify = np.zeros(N_reduced, dtype=bool)

            num_to_amplify = int(m - np.sum(keep))
            chosen = rng.choice(candidate_inds, size=min(num_to_amplify, candidate_inds.size), replace=False, p=probs[candidate_inds]/np.sum(probs[candidate_inds]))
            amplify[chosen] = True

            # Normalize amplified entries
            with np.errstate(divide='ignore', invalid='ignore'):
                reduced[amplify] = np.divide(reduced[amplify], probs[amplify], where=probs[amplify] > 0)

            reduced[~(keep | amplify)] = 0
        else:
            reduced[~keep] = 0

        result[nonzeros] = reduced

    return result

def hthresholding(v, m):

    temp = np.sort(np.abs(v))

    thresh = temp[len(v) - m]

    np.putmask(v, abs(v) < thresh, [0])

    return v