import numpy as np
import tensorflow as tf
import pennylane as qml
from tensorflow.keras.models import load_model
import os

class QuatideEngine:
    def __init__(self, model_path='models/quatide_model.h5'):
        self.n_qubits = 4
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.model_path = model_path
        self.model = self._load_hybrid_model()

    def _load_hybrid_model(self):
        # في بيئة الإنتاج، نحمل النموذج المدرب مسبقاً
        if os.path.exists(self.model_path):
            return load_model(self.model_path, custom_objects={'tf': tf})
        return None

    def quantum_node(self, inputs):
        @qml.qnode(self.dev, interface="tf")
        def circuit(inputs):
            for i, val in enumerate(inputs[:self.n_qubits]):
                qml.RY(val, wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        return circuit(inputs)

    def predict_waste(self, lstm_input_data, raw_features):
        """
        توقع الفضلات بناءً على البيانات القادمة من الـ API
        """
        # تحويل الميزات لبيانات كمية
        q_feat = self.quantum_node(np.mean(raw_features, axis=0))
        q_feat = tf.reshape(q_feat, (1, 1))
        
        # التوقع
        prediction = self.model.predict([lstm_input_data, q_feat])
        return float(prediction[0][0])

