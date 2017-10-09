import logging


class XLogger:
    def __init__(self, log_path, full_format=True):
        format = '%(asctime)s %(name)-12s %(levelname)-8s \n%(message)s\n' if full_format else '%(message)s'
        logging.basicConfig(level=logging.DEBUG,
                            format=format,
                            datefmt='%m-%d %H:%M',
                            filename=log_path,
                            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(format))
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

    @staticmethod
    def get():
        return logging
