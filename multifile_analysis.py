from multifile_feature import MultifileFeatureExtractor
from multifile_lda import MultifileLDA

class MultifileAnalysis(object):

    def load_data(self, input_set, scaling_factor=100, normalise=0,
                 fragment_grouping_tol=7, loss_grouping_tol=10, 
                 loss_threshold_min_count=15, loss_threshold_max_val=200, 
                 input_type='filename'):

        self.F = len(input_set)
        self.dfs = {}
        self.ms1s = {}
        self.ms2s = {}
        self.Ds = {}
        self.vocab = None        
        
        extractor = MultifileFeatureExtractor(input_set, fragment_grouping_tol, loss_grouping_tol, 
                                              loss_threshold_min_count, loss_threshold_max_val, 
                                              input_type)
        
        fragment_q = extractor.make_fragment_queue()
        fragment_groups = extractor.group_features(fragment_q, extractor.fragment_grouping_tol)

        loss_q = extractor.make_loss_queue()
        loss_groups = extractor.group_features(loss_q, extractor.loss_grouping_tol, check_threshold=True)
        extractor.create_dataframes(fragment_groups, loss_groups)
            
        for f in range(self.F):        

            extractor.normalise(f, scaling_factor)
            df, vocab, ms1, ms2 = extractor.get_entry(f)
                
            nrow, ncol = df.shape
            assert nrow == len(ms1), "df shape %s doesn't match %d" % (df.shape, len(ms1))
            assert ncol == len(vocab)

            self.Ds[f] = nrow
            self.dfs[f] = df
            self.ms1s[f] = ms1
            self.ms2s[f] = ms2
            self.vocab = vocab
            
    def load_synthetic(self, dfs, vocab):

        self.F = len(dfs)
        self.dfs = dfs
        self.vocab = vocab
        self.Ds = {}
        
        for f in range(self.F):        
            df = self.dfs[f]
            nrow, _ = df.shape
            self.Ds[f] = nrow
            
    def run(self, K, alpha, beta, n_burn=100, n_samples=200, n_thin=0):

        lda = MultifileLDA(self.dfs, self.vocab)
        lda.run(K, alpha, beta, n_burn, n_samples, n_thin)
        return lda