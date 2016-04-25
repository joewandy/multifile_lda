from multifile_feature import FeatureExtractor
from multifile_lda import MultifileLDA
import multifile_utils as utils
import numpy as np
from astropy.modeling.core import Model

class MultifileAnalysis(object):
    
    def __init__(self):
        
        self.F = 0
        self.K = 0
        self.counts = {}
        self.ms1s = {}
        self.ms2s = {}
        self.Ds = {}
        self.vocab = None        

    def load_data(self, input_set, scaling_factor=100, normalise=0,
                 fragment_grouping_tol=7, loss_grouping_tol=10, 
                 loss_threshold_min_count=15, loss_threshold_max_val=200, 
                 input_type='filename'):

        self.F = len(input_set)        
        extractor = FeatureExtractor(input_set, fragment_grouping_tol, loss_grouping_tol, 
                                     loss_threshold_min_count, loss_threshold_max_val, 
                                     input_type)
        
        fragment_q = extractor.make_fragment_queue()
        fragment_groups = extractor.group_features(fragment_q, extractor.fragment_grouping_tol)

        loss_q = extractor.make_loss_queue()
        loss_groups = extractor.group_features(loss_q, extractor.loss_grouping_tol, check_threshold=True)
        extractor.create_counts(fragment_groups, loss_groups)
            
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
            for f in range(len(self.doc_topic_)):
                self.thresholded_doc_topic.append(utils.threshold_matrix(self.model.doc_topic_[f], epsilon=th_doc_topic))        
        else:
            self.thresholded_doc_topic.append(utils.threshold_matrix(self.model.doc_topic_, epsilon=th_doc_topic))        
            
    def get_top_words(self, with_probabilities=True, selected=None, verbose=True):
        
        topic_words_map = {}
        for i, topic_dist in enumerate(self.thresholded_topic_word):
            
            if selected is not None and i not in selected:
                continue
            
            ordering = np.argsort(topic_dist)
            topic_words = np.array(self.vocab)[ordering][::-1]
            dist = topic_dist[ordering][::-1]        
            topic_name = 'Topic {}:'.format(i)
            
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
            
            topic_words_map[i] = (topic_words, dist)
        
        return topic_words_map

    def get_top_docs(self, f, with_probabilities=True, selected=None, verbose=True):

        ms1_peakid = self.ms1s[f]['peakID'].values        
        topic_docs_map = {}
        for i, topic_dist in enumerate(self.thresholded_doc_topic[f].transpose()):
            
            if selected is not None and i not in selected:
                continue
            
            ordering = np.argsort(topic_dist)
            topic_docs = np.array(ms1_peakid)[ordering][::-1]
            dist = topic_dist[ordering][::-1]        
            topic_name = 'Topic {}:'.format(i)
            
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
            
            topic_docs_map[i] = (topic_docs, dist)
        
        return topic_docs_map
