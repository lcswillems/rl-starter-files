import time
import datetime
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from model import ACModel


def train(args):
# if __name__ == "__main__":

    args['mem'] = args.get('recurrence') > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.get('env')}_{args.get('algo')}_seed{args.get('seed')}_{date}"

    model_name = args.get('model') or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments #NOTE: see if this removes unneeded output
    # txt_logger.info("{}\n".format(" ".join(sys.argv)))
    # txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.get('seed'))

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = args.get('envs')

    # envs = []
    # for i in range(args.get('procs')):
    #     envs.append(utils.make_env(args.get('env'), args.get('seed') + 10000 * i))
    # txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.get('mem'), args.get('text'))
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.get('algo') == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.get('frames_per_proc'), args.get('discount'), args.get('lr'), args.get('gae_lambda'),
                                args.get('entropy_coef'), args.get('value_loss_coef'), args.get('max_grad_norm'), args.get('recurrence'),
                                args.get('optim_alpha'), args.get('optim_eps'), preprocess_obss)
    elif args.get('algo') == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.get('frames_per_proc'), args.get('discount'), args.get('lr'), args.get('gae_lambda'),
                                args.get('entropy_coef'), args.get('value_loss_coef'), args.get('max_grad_norm'), args.get('recurrence'),
                                args.get('optim_eps'), args.get('clip_eps'), args.get('epochs'), args.get('batch_size'), preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.get('algo')))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.get('frames'):
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.get('log_interval') == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.get('save_interval') > 0 and update % args.get('save_interval') == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")


