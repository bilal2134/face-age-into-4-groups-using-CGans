# app.py
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

# Import necessary Keras layers
from tensorflow.keras import layers


# Define the custom AttentionBlock class
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.gamma = tf.Variable(0.0, trainable=True)
        self.query_conv = layers.Conv2D(filters // 8, 1, padding='same')
        self.key_conv = layers.Conv2D(filters // 8, 1, padding='same')
        self.value_conv = layers.Conv2D(filters, 1, padding='same')

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Compute query, key, value
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        # Reshape for attention computation
        query_reshaped = tf.reshape(query, [batch_size, -1, self.filters // 8])
        key_reshaped = tf.reshape(key, [batch_size, -1, self.filters // 8])
        value_reshaped = tf.reshape(value, [batch_size, -1, self.filters])

        # Compute attention scores
        attention = tf.matmul(query_reshaped, key_reshaped, transpose_b=True)
        attention = tf.nn.softmax(attention, axis=-1)

        # Apply attention to value
        context = tf.matmul(attention, value_reshaped)
        context = tf.reshape(context, [batch_size, height, width, self.filters])

        # Apply gamma and residual connection
        output = self.gamma * context + inputs

        return output


# Load the trained generator model
# Load the trained generator model
generator = tf.keras.models.load_model(
    'final_generator (2).keras',
    custom_objects={'AttentionBlock': AttentionBlock},
    compile=False
)


# Define age categories
AGE_CATEGORIES = ['Child', 'Young', 'Middle-aged', 'Old']

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get selected age category from the form
        age_category = request.form.get('age_category')
        if age_category not in AGE_CATEGORIES:
            age_category = 'Young'  # Default category

        # Generate image
        generated_image = generate_image(age_category)

        # Convert image to base64 string for display in HTML
        img_io = io.BytesIO()
        generated_image.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')

        return render_template('index.html', img_data=img_base64, selected_age=age_category,
                               age_categories=AGE_CATEGORIES)
    else:
        # Render the form for the first time
        return render_template('index.html', img_data=None, selected_age=None, age_categories=AGE_CATEGORIES)


def generate_image(age_category):
    # Map age category to label index
    label_index = AGE_CATEGORIES.index(age_category)

    # Generate random noise
    noise = tf.random.normal([1, 100])

    # Create one-hot label
    label = tf.one_hot([label_index], depth=4)

    # Generate image using the generator model
    generated_image = generator([noise, label], training=False)

    # Convert tensor to image
    generated_image = (generated_image[0] + 1) / 2.0  # Rescale to [0, 1]
    generated_image = tf.clip_by_value(generated_image, 0.0, 1.0)
    generated_image = (generated_image.numpy() * 255).astype(np.uint8)
    image = Image.fromarray(generated_image)

    return image


if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
