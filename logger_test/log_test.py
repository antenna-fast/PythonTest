import logging


def create_logger(log_file_path='default_log_name.log', log_level=1, is_overwrite=1):
    # create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    # to log file
    # if is_overwrite:
    file_handler = logging.FileHandler(log_file_path, 'w' if is_overwrite else 'a')
    # else:
    #     file_handler = logging.FileHandler(log_file_path, 'a')
    file_handler.setLevel(logging.INFO)

    # to terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # output format
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


seg_line = '* ' + '-' * 20 + ' *'


def para_seg_line(param):
    return '* ' + '-  ' * 10 + str(param) + '  -' * 10 + ' *'


if __name__ == '__main__':
    logger = create_logger()

    logger.info('START training ... ')

    MAX_EPOCH = 10
    BATCH_NUM = 10
    for epoch in range(MAX_EPOCH):
        for batch in range(BATCH_NUM):
            logger.info('EPOCH:[{}/{}] BATCH:[{}/{}] loss={:.4f} acc={:.4f}'.
                        format(epoch, MAX_EPOCH, batch, BATCH_NUM, 0, 0))

        logger.info('VAL: {} loss={:.4f} acc={:.4f}'.format(epoch, 0, 0))
        logger.info('TEST: {} loss={:.4f} acc={:.4f}'.format(epoch, 0, 0))

