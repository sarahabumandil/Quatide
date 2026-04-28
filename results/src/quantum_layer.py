import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def circuit(inputs):
    """
    Quantum circuit for feature encoding.
    Args:
        inputs: A tensor of 4 features.
    """
    # Angle Encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
        
    return qml.expval(qml.PauliZ(0))

def quantum_feature_mapping(feature_matrix):
    """
    Wraps the QNode to process a batch of features.
    Returns a (samples, 1) tensor.
    """
    # Ensure input is 4-dimensional for the 4 qubits
    if feature_matrix.shape[1] < n_qubits:
        padding = tf.zeros((feature_matrix.shape[0], n_qubits - feature_matrix.shape[1]))
        feature_matrix = tf.concat([feature_matrix, padding], axis=1)
    elif feature_matrix.shape[1] > n_qubits:
        feature_matrix = feature_matrix[:, :n_qubits]

    # Map the QNode over the batch
    q_results = [circuit(x) for x in feature_matrix]
    return tf.reshape(tf.convert_to_tensor(q_results, dtype=tf.float32), (-1, 1))
  
