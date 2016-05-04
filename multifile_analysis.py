import timeit
import time
import gzip
import cPickle

import pandas as pd
import seaborn as sns
import numpy as np
from IPython.display import display, HTML

from multifile_feature import SparseFeatureExtractor
from multifile_lda import MultifileLDA
import multifile_utils as utils

class MultifileAnalysis(object):

    def __init__(self):

        self.F = 0
        self.K = 0
        self.counts = {}
        self.ms1s = {}
        self.ms2s = {}
        self.Ds = {}
        self.vocab = None

    def load_data(self, input_set, scaling_factor,
                fragment_grouping_tol, loss_grouping_tol,
                loss_threshold_min_count, loss_threshold_max_val,
                normalise=0, input_type='filename'):

        self.F = len(input_set)
        extractor = SparseFeatureExtractor(input_set, fragment_grouping_tol, loss_grouping_tol,
                                     loss_threshold_min_count, loss_threshold_max_val,
                                     input_type=input_type)

        fragment_q = extractor.make_fragment_queue()
        fragment_groups = extractor.group_features(fragment_q, extractor.fragment_grouping_tol)

        loss_q = extractor.make_loss_queue()
        loss_groups = extractor.group_features(loss_q, extractor.loss_grouping_tol, check_threshold=True)
        extractor.create_counts(fragment_groups, loss_groups, scaling_factor)

        for f in range(self.F):

            count, vocab, ms1, ms2 = extractor.get_entry(f)
            nrow, ncol = count.shape
            assert nrow == len(ms1), "count shape %s doesn't match %d" % (count.shape, len(ms1))
            assert ncol == len(vocab)

            self.Ds[f] = nrow
            self.counts[f] = count
            self.ms1s[f] = ms1
            self.ms2s[f] = ms2
            self.vocab = vocab

    def load_synthetic(self, counts, vocab, model):

        self.F = len(counts)
        self.counts = counts
        self.vocab = vocab
        self.model = model
        self.K = self.model.K

        for f in range(self.F):
            df = self.counts[f]
            nrow, _ = df.shape
            self.Ds[f] = nrow

    def set_data(self, counts, vocab, ms1s, ms2s, model):

        self.F = len(counts)
        self.vocab = vocab
        self.model = model
        self.K = self.model.K
        for f in range(self.F):

            count = counts[f]
            ms1 = ms1s[f]
            ms2 = ms2s[f]
            nrow, ncol = count.shape
            assert nrow == len(ms1), "count shape %s doesn't match %d" % (count.shape, len(ms1))
            assert ncol == len(vocab)

            self.Ds[f] = nrow
            self.counts[f] = count
            self.ms1s[f] = ms1
            self.ms2s[f] = ms2

    def run(self, K, alpha, beta, n_burn=100, n_samples=200, n_thin=0):

        lda = MultifileLDA(self.counts, self.vocab)
        lda.run(K, alpha, beta, n_burn, n_samples, n_thin)
        self.model = lda
        self.K = K
        return lda

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

    def do_thresholding(self, th_doc_topic=0.05, th_topic_word=0.1):

        # save the thresholding values used for visualisation later
        self.th_doc_topic = th_doc_topic
        self.th_topic_word = th_topic_word

        # get rid of small values in the matrices of the results
        # if epsilon > 0, then the specified value will be used for thresholding
        # otherwise, the smallest value for each row in the matrix is used instead
        self.thresholded_topic_word = utils.threshold_matrix(self.model.topic_word_, epsilon=th_topic_word)
        self.thresholded_doc_topic = []
        if type(self.model.doc_topic_) is list:
            for f in range(len(self.model.doc_topic_)):
                self.thresholded_doc_topic.append(utils.threshold_matrix(self.model.doc_topic_[f], epsilon=th_doc_topic))
        else:
            self.thresholded_doc_topic.append(utils.threshold_matrix(self.model.doc_topic_, epsilon=th_doc_topic))

    def get_top_words(self, with_probabilities=True, selected=None, verbose=True, limit=None):

        topic_words_map = {}
        for k, topic_dist in enumerate(self.thresholded_topic_word):

            if selected is not None and k not in selected:
                continue

            ordering = np.argsort(topic_dist)
            topic_words = np.array(self.vocab)[ordering][::-1]
            dist = topic_dist[ordering][::-1]
            topic_name = 'Topic {}:'.format(k)

            # TODO: vectorise this
            if verbose:
                print topic_name,
                for j in range(len(topic_words)):
                    if dist[j] > 0:
                        if with_probabilities:
                            print '%s (%.3f),' % (topic_words[j], dist[j]),
                        else:
                            print('{},'.format(topic_words[j])),
                    else:
                        break
                print
                print
            else:
                for j in range(len(topic_words)):
                    if dist[j] == 0:
                        break

            if limit is not None and limit < j:
                j = limit
            topic_words = topic_words[:j]
            dist = dist[:j]
            topic_words_map[k] = (topic_words, dist)

        return topic_words_map

    def get_top_docs(self, f, with_probabilities=True, selected=None, verbose=True, limit=None):

        ms1_peakid = self.ms1s[f]['peakID'].values
        topic_docs_map = {}
        for k, topic_dist in enumerate(self.thresholded_doc_topic[f].transpose()):

            if selected is not None and k not in selected:
                continue

            ordering = np.argsort(topic_dist)
            topic_docs = np.array(ms1_peakid)[ordering][::-1]
            dist = topic_dist[ordering][::-1]
            topic_name = 'Topic {}:'.format(k)

            # TODO: vectorise this
            if verbose:
                print topic_name,
                for j in range(len(topic_docs)):
                    if dist[j] > 0:
                        if with_probabilities:
                            print '%s (%.3f),' % (topic_docs[j], dist[j]),
                        else:
                            print('{},'.format(topic_docs[j])),
                    else:
                        break
                print
                print
            else:
                for j in range(len(topic_docs)):
                    if dist[j] == 0:
                        break

            if limit is not None and limit < j:
                j = limit
            topic_docs = topic_docs[:j]
            dist = dist[:j]
            topic_docs_map[k] = (topic_docs, dist)

        return topic_docs_map

    def get_rankings(self, interesting=None, min_degree=0):

        if interesting is None:
            interesting = [k for k in range(self.K)]

        file_ids = []
        topic_ids = []
        degrees = []
        h_indices = []
        doc_citations = []
        for f in range(self.F): # for each file

            file_ids.extend([f for k in range(self.K)])
            topic_ids.extend([k for k in range(self.K)])

            # compute the degrees of all topics
            doc_topic = self.thresholded_doc_topic[f]
            columns = (doc_topic>0).sum(0)
            assert len(columns) == self.K
            degrees.extend(columns)

            # compute the h-indices of all topics
            columns, citations = self._h_index(f)
            h_indices.extend(columns)
            doc_citations.append(citations)

        rows = []
        for i in range(len(topic_ids)):
            topic_id = topic_ids[i]
            if topic_id in interesting and degrees[i]>min_degree:
                rows.append((file_ids[i], topic_id, degrees[i], h_indices[i]))

        df = pd.DataFrame(rows, columns=['file', 'M2M', 'degree', 'h-index'])
        return df, doc_citations

    # compute the h-index of topics TODO: this only works for fragment and loss words!
    def _h_index(self, f):

        h_indices = []
        doc_citations = {}
        topic_words_map = self.get_top_words(verbose=False)
        topic_docs_map = self.get_top_docs(f, verbose=False)

        ks = range(self.K)
        for k in ks:

            # find the top words and documents in this topic above the threshold
            top_words, _ = topic_words_map[k]
            top_docs, _ = topic_docs_map[k]

            topic_words = {}
            for word in top_words:
                topic_words[word] = 0

            # handle empty topics
            if len(top_docs) == 0:
                h_indices.append(0)
                continue
            else:

                # now find out how many of the documents in this topic actually 'cite' the words
                for parent_peakid in top_docs:

                    # find all the fragment peaks of this parent peak
                    ms2_rows = self.ms2s[f].loc[self.ms2s[f]['MSnParentPeakID']==parent_peakid]
                    fragment_bin_ids = ms2_rows[['fragment_bin_id']]
                    loss_bin_ids = ms2_rows[['loss_bin_id']]

                    # convert from pandas dataframes to list
                    fragment_bin_ids = fragment_bin_ids.values.ravel().tolist()
                    loss_bin_ids = loss_bin_ids.values.ravel().tolist()

                    # convert to set for quick lookup
                    word_set = set()
                    for bin_id in fragment_bin_ids:
                        new_word = 'fragment_%s' % bin_id
                        word_set.add(new_word)
                    for bin_id in loss_bin_ids:
                        new_word = 'loss_%s' % bin_id
                        word_set.add(new_word)

                    # count the citation numbers
                    doc_citing = 0
                    for word in topic_words:
                        if word in word_set:
                            topic_words[word] += 1
                            doc_citing += 1
                    doc_citations[(k, parent_peakid)] = doc_citing

                    # make a dataframe of the articles & citation counts
                    df = pd.DataFrame(topic_words, index=['counts']).transpose()
                    df = df.sort(['counts'], ascending=False)

                    # compute the h-index
                    h_index = 0
                    for index, row in df.iterrows():
                        if row['counts'] > h_index:
                            h_index += 1
                        else:
                            break

                print 'k=%d h-index=%d' % (k, h_index)
                h_indices.append(h_index)

        return h_indices, doc_citations
