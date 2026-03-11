from utils.sit_transport import build_sit_training_batch


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    del train_state
    return build_sit_training_batch(
        path_type=FLAGS.model['transport_path_type'],
        prediction=FLAGS.model['transport_prediction'],
        loss_weight=FLAGS.model['transport_loss_weight'],
        train_eps=FLAGS.model['transport_train_eps'],
        sample_eps=FLAGS.model['transport_sample_eps'],
        dataset_name=FLAGS.dataset_name,
        denoise_timesteps=FLAGS.model['denoise_timesteps'],
        class_dropout_prob=FLAGS.model['class_dropout_prob'],
        num_classes=FLAGS.model['num_classes'],
        key=key,
        images=images,
        labels=labels,
        force_t=force_t,
        force_dt=force_dt,
    )
