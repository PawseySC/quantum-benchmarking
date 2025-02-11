# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 10:13 2025

@author: james
"""

import unittest
import jax.numpy as jnp
import jax.random as jrandom

import qnnax

class ImportTests(unittest.TestCase):
    def test_qnn_imports(self):
        self.assertIsNotNone(qnnax.QNN)
        self.assertIsNotNone(qnnax.DenseQNN)
        self.assertIsNotNone(qnnax.ReuploaderQNN)

    def test_dataset_imports(self):
        self.assertIsNotNone(qnnax.create_moons)
        self.assertIsNotNone(qnnax.create_blobs)
        self.assertIsNotNone(qnnax.create_circles)
        self.assertIsNotNone(qnnax.create_classification)


class DenseQNNTests(unittest.TestCase):
    def test_transforms(self):
        key = jrandom.PRNGKey(0)

        X = jrandom.uniform(key, shape=(100, 8), minval=-3, maxval=-1, dtype=jnp.float32)
        qnn = qnnax.DenseQNN(num_qubits=3)
        X_ = qnn.transform(X)
        self.assertEqual(X_.shape, X.shape)
        self.assertEqual(X_.dtype, jnp.float32)
        self.assertLessEqual(X_.max(), 1)
        self.assertGreaterEqual(X_.min(), 0)
        self.assertAlmostEqual(jnp.sum(X_**2).item(), 100, delta=0.0001)

        X = jrandom.uniform(key, shape=(77, 3), minval=-3, maxval=-1, dtype=jnp.float32)
        qnn = qnnax.DenseQNN(num_qubits=3)
        X_ = qnn.transform(X)
        self.assertNotEqual(X_.shape, X.shape)
        self.assertEqual(X_.shape, (77, 8))
        self.assertEqual(X_.dtype, jnp.float32)
        self.assertLessEqual(X_.max(), 1)
        self.assertGreaterEqual(X_.min(), 0)
        self.assertAlmostEqual(jnp.sum(X_**2).item(), 77, delta=0.0001)


class ReuploaderQNNTests(unittest.TestCase):
    def test_transforms(self):
        key = jrandom.PRNGKey(0)

        X = jrandom.uniform(key, shape=(100, 3), minval=-123, maxval=33, dtype=jnp.float32)
        qnn = qnnax.ReuploaderQNN(num_features=3)
        X_ = qnn.transform(X)
        self.assertEqual(X_.shape, X.shape)
        self.assertEqual(X_.dtype, jnp.float32)

        X = jrandom.uniform(key, shape=(77, 4), minval=-3, maxval=-1, dtype=jnp.float32)
        qnn = qnnax.ReuploaderQNN(num_features=4)
        X_ = qnn.transform(X)
        self.assertNotEqual(X_.shape, X.shape)
        self.assertEqual(X_.shape, (77, 6))
        self.assertEqual(X_.dtype, jnp.float32)

        X = jrandom.uniform(key, shape=(77, 2), minval=-3, maxval=-1, dtype=jnp.float32)
        qnn = qnnax.ReuploaderQNN(num_features=4)
        X_ = qnn.transform(X)
        self.assertNotEqual(X_.shape, X.shape)
        self.assertEqual(X_.shape, (77, 6))
        self.assertEqual(X_.dtype, jnp.float32)


if __name__ == '__main__':
    unittest.main()
