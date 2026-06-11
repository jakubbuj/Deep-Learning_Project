from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def get_callbacks(model_name):
    """ 
    Defines standard, necessary callbacks for consistent training across all models. 
    
    Args:
        model_name (str): A unique identifier for the model (e.g., 'custom_cnn', 'resnet50_ft').
                          Used to name the saved weights file.
                          
    Returns:
        list: A list of Keras Callback objects.
    """
    
    # 1. Model Checkpoint: Saves weights only when validation loss improves.

    checkpoint = ModelCheckpoint(
        filepath=f'results/best_model_weights_{model_name}.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True, # Only save weights to keep file size down
        verbose=1
    )
    
    # 2. Early Stopping: Stops training if validation loss doesn't improve.

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # Reverts model to the best state seen during training
        verbose=1
    )
    
    # 3. Reduce Learning Rate on Plateau: Gently adjusts the learning rate.

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Reduce learning rate by 80%
        patience=5, # Wait 5 epochs before reducing
        min_lr=1e-6, # Do not go below this rate
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def compile_model(model, learning_rate=1e-4):
    """ 
    Compiles the model with a standard optimizer, loss function, and metrics. 
    
    Args:
        model (keras.Model): The built model architecture (from src/models.py).
        learning_rate (float): The initial learning rate (important to keep small for fine-tuning).
        
    Returns:
        keras.Model: The compiled model instance.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        # Use categorical_crossentropy because your generator uses class_mode='categorical' (one-hot encoding)
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model