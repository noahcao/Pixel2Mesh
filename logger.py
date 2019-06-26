import logging
import os


def create_logger(cfg, phase='train'):
    log_file = '{}_{}.log'.format(cfg.version, phase)
    final_log_file = os.path.join(cfg.log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
