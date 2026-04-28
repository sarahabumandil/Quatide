from tensorflow.keras import layers, Model, Input

def build_quantum_hybrid_model(timesteps, n_features):
    """
    Constructs a hybrid LSTM-Quantum Keras model.
    """
    # Classical Path (LSTM)
    lstm_input = Input(shape=(timesteps, n_features), name="ts_input")
    x = layers.LSTM(64, return_sequences=False)(lstm_input)
    x = layers.Dropout(0.2)(x)

    # Quantum Input Path
    quantum_input = Input(shape=(1,), name="q_input")
    
    # Merging
    combined = layers.Concatenate()([x, quantum_input])
    
    # Dense Head
    z = layers.Dense(32, activation='relu')(combined)
    output = layers.Dense(1, activation='linear', name="output")(z)

    model = Model(inputs=[lstm_input, quantum_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

