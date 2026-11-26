import unittest
import numpy as np
from metrics.definitions import (correlation,emotion_alignment,color_timbre_alignment,
    chroma_diversity,harmonic_consistency,dynamic_variation)


class MetricDefinitionTests(unittest.TestCase):
    def test_eas_is_rank_correlation_with_shared_lag(self):
        x=np.array([[1,6],[2,5],[3,4],[4,3],[5,2],[6,1]],float)
        y=x**4
        value,lag=emotion_alignment(x,y,sample_hz=2,max_lag_seconds=0)
        self.assertAlmostEqual(value,1.)
        self.assertEqual(lag,0)
        self.assertLess(correlation(x[:,0],y[:,0]),.95)

    def test_cta_sign_and_undefined_constant(self):
        x=np.array([0,1,2,3],float)
        self.assertAlmostEqual(color_timbre_alignment(x,x,2*x,-x),1.)
        with self.assertRaises(ValueError):color_timbre_alignment(x,x,x,np.zeros(4))

    def test_chroma_is_pairwise_cosine_not_entropy(self):
        x=np.zeros((2,12,8));x[:,0]=1
        self.assertAlmostEqual(chroma_diversity(x),0.)
        x[1,0]=0;x[1,7]=1
        self.assertAlmostEqual(chroma_diversity(x),1.)
        with self.assertRaises(ValueError):chroma_diversity(x[:1])

    def test_hcs_has_explicit_scale_and_no_clamping(self):
        tc=np.zeros((3,6));tc[:,0]=[0,2,4]
        self.assertEqual(harmonic_consistency(tc,sigma_tc=1),-1)
        self.assertEqual(harmonic_consistency(tc,sigma_tc=4),.5)
        self.assertEqual(harmonic_consistency(np.ones((3,6)),sigma_tc=1),1)
        with self.assertRaises(ValueError):harmonic_consistency(tc,sigma_tc=0)

    def test_dv_is_aligned_db_cosine(self):
        self.assertAlmostEqual(dynamic_variation([-20,-40,-60],[-20,-40,-60]),0)
        self.assertGreater(dynamic_variation([-20,-40,-60],[-60,-40,-20]),0)
        with self.assertRaises(ValueError):dynamic_variation([-20,-30],[-20,-30,-40])


if __name__=='__main__':unittest.main()
