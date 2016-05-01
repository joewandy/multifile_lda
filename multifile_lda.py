import cPickle
import gzip
import sys
import time
import timeit

from numpy import int32
from numpy.random import RandomState

from multifile_cgs_numpy import sample_numpy
import multifile_utils as utils
import numpy as np

# from scipy.sparse import lil_matrix
# from multifile_cgs_numba import sample_numba

class MultifileLDA(object):

    def __init__(self, dfs, vocab, random_state=None):

        # make sure to get the same results from running gibbs each time
        if random_state is None:
            self.random_state = RandomState(1234567890)
        else:
            self.random_state = random_state

        self.F = len(dfs)
        self.dfs = dfs
        self.vocab = vocab
        self.Ds = {}

        for f in range(self.F):
            df = self.dfs[f]
            nrow, _ = df.shape
            self.Ds[f] = nrow

    def run(self, K, alpha, beta, n_burn, n_samples, n_thin):

        self.K = K
        self.N = len(self.vocab)

        # beta is shared across all files
        self.beta = np.ones(self.N) * beta

        # set the matrices for each file
        self.alphas = {}
        self.ckn = {}
        self.ck = {}
        self.cdk = {}
        self.cd = {}
        for f in range(self.F):
            self.alphas[f] = np.ones(self.K) * alpha
            self.ckn[f] = np.zeros((self.K, self.N), dtype=int32)
            self.ck[f] = np.zeros(self.K, dtype=int32)
            self.cdk[f] = np.zeros((self.Ds[f], self.K), int32)
            self.cd[f] = np.zeros(self.Ds[f], int32)

        # randomly assign words to topics
        # also turn word counts in the document into a vector of word occurences
        print "Initialising "
        self.Z = {}
        self.document_indices = {}
        for f in range(self.F):
            print " - file " + str(f) + " ",
            for d in range(self.Ds[f]):
                if d%10==0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                document = self.dfs[f].iloc[[d]]
                word_idx = utils.word_indices(document)
                word_locs = []
                for pos, n in enumerate(word_idx):
                    k = self.random_state.randint(self.K)
                    file_cdk = self.cdk[f]
                    file_cd = self.cd[f]
                    file_ckn = self.ckn[f]
                    file_ck = self.ck[f]
                    file_cdk[d, k] += 1
                    file_cd[d] += 1
                    file_ckn[k, n] += 1
                    file_ck[k] += 1
                    self.Z[(f, d, pos)] = k
                    word_locs.append((pos, n))
                self.document_indices[(f, d)] = word_locs
            print
        print

        # select the sampler function to use
        sampler_func = None
        try:
            # print "Using Numba for multi-file LDA sampling"
            # sampler_func = sample_numba
            print "Using Numpy for multi-file LDA sampling"
            sampler_func = sample_numpy
        except Exception:
            print "Using Numpy for multi-file LDA sampling"
            sampler_func = sample_numpy

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.topic_word_, self.doc_topic_, self.posterior_alphas, self.log_likelihoods = sampler_func(
                self.random_state, n_burn, n_samples, n_thin,
                self.F, self.Ds, self.N, self.K, self.document_indices,
                self.alphas, self.beta, self.Z,
                self.cdk, self.cd, self.ckn, self.ck)

    @classmethod
    def resume_from(cls, project_in):
        start = timeit.default_timer()
        with gzip.GzipFile(project_in, 'rb') as f:
            obj = cPickle.load(f)
            stop = timeit.default_timer()
            print "Project loaded from " + project_in + " time taken = " + str(stop-start)
            return obj

    def save_project(self, project_out, message=None):
        start = timeit.default_timer()
        self.last_saved_timestamp = str(time.strftime("%c"))
        self.message = message
        with gzip.GzipFile(project_out, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            stop = timeit.default_timer()
            print "Project saved to " + project_out + " time taken = " + str(stop-start)
