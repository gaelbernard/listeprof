import os, logging, duckdb
from abc import ABC, abstractmethod

class OperationAbstract(ABC):
    def __init__(self, db_path: str):
        self.con_path = db_path
        self.con = duckdb.connect(db_path)
        self.sample = True

    def pre_abstract(self):
        """Always run before pre()."""
        logging.info(f"Operation {self.__class__.__name__} starting.")

    def pre(self):
        """Pre checks and setup to be implemented in subclasses."""
        ...

    def trans(self):
        """Main transformation step to be implemented in subclasses."""
        ...

    def post_abstract(self):
        logging.info(f"Operation {self.__class__.__name__} completed.")

    def run(self):
        try:
            self.pre_abstract()
            self.con.execute("BEGIN")
            self.pre()
            self.trans()
            self.con.execute("COMMIT")
            self.post_abstract()
        except Exception as e:
            self.con.execute("ROLLBACK")
            self._quit_on_failure(f'Pipeline failed: {e}')
            raise
        finally:
            if self.con:
                self.con.close()

    def _quit_on_failure(self, error_msg=None):
        if error_msg:
            logging.error(error_msg)
        if self.con:
            self.con.close()
        logging.error("Quitting due to failure.")
        raise SystemExit(1)
