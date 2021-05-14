import numpy as np

def generate_real_samples(dataset, n_samples: int):
    """
    Randomly samples from the dataset
    """

    # Load data from model
    num_points = len(dataset.notes) - dataset.sequence_len + 1

    # Randomly sample points from dataset
    data_idxs = np.random.randint(0, num_points, n_samples)

    # Get the data
    X = np.array([dataset.get_idx(i) for i in data_idxs])
    X = np.expand_dims(X, 3)
    y = np.ones((n_samples, 1))

    return X, y


def generate_fake_samples(model, latent_dim, n_samples):
    """
    Randomly generates fake data
    """
    # Generate input to generator
    x_in = generate_latent_points(latent_dim, n_samples)

    # Predict outputs
    X = model.predict(x_in)
    y = np.zeros((n_samples, 1))

    return X, y


def generate_latent_points(latent_dim, n_samples):
    """
    Generates vectors from the latent space
    """
    return np.random.normal(size=(n_samples, latent_dim))


def train_gan(generator, discriminator, gan, dataset,
              callbacks = None,
              latent_dims: int = 100, n_epochs: int = 100,
              batch_size: int = 32, chkpt_dist: int = 100,
              base_path: str = None, test_name: str = None):
    # Figure out number of batches
    num_batches = int(len(dataset.notes) / batch_size)

    # Iterate through the epochs
    for e in range(n_epochs):
        # Iterate over batches
        for b in range(num_batches):
            # Generate data for discriminator
            X_real, y_real = generate_real_samples(dataset, batch_size)
            X_fake, y_fake = generate_fake_samples(generator, latent_dims, batch_size)

            X_dis, y_dis = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            # Train discriminator
            d_loss = discriminator.train_on_batch(X_dis, y_dis, callbacks=callbacks)

            # Generate data for generator
            X_gan = generate_latent_points(latent_dims, batch_size)
            y_gan = np.ones((batch_size, 1))

            # Train generator (with frozen generator)
            g_loss = gan.train_on_batch(X_gan, y_gan, callbacks=callbacks)

            # Checkpoint
            if b % chkpt_dist == 0:
                print(f'Epoch {e}, Batch {b+1}/{num_batches}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')
                gen_path = f'{base_path}{test_name}_gen.hdf5'
                dis_path = f'{base_path}{test_name}_dis.hdf5'
                generator.save(gen_path)
                discriminator.save(dis_path)
