import wandb


class Logger:
    def __init__(self, project_name, config, enabled=True):
        self.enabled = enabled
        if self.enabled:
            wandb.init(project=project_name, config=config)

    def log_dict(self, metrics, step=None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()