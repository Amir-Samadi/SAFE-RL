import os

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image
from matplotlib import pyplot as plt
import keras

import olson.model as olson_model
from dataset_evaluation import _get_subfolders
from star_gan.model import Generator
from util import get_action_names, restrict_tf_memory, get_agent_action, generate_counterfactual, \
    generate_olson_counterfactual, load_olson_models, load_baselines_model


class Evaluator:
    def __init__(self, agent, test_set_path, env_name, agent_type="deepq", img_size=176, ablate_agent=False):
        """
        Provides functionality for quantitative evaluations of counterfactual explainability approaches on the given
        test set.

        :param agent: The agent that was used to generate the data set.
        :param test_set_path: Path to the test set that should be used for the evaluation.
        :param env_name: Name of the Gym environment that was used to generate the data set.
        :param img_size: Size of frames from the data set (Quadratic images are assumed).
        :param agent_type: The type of agent. "deepq" for keras deepq, "acer" for baselines acer, "torch" for a pytorch acer-critic
        """
        self.agent = agent
        self.test_set_path = test_set_path
        self.img_size = img_size
        self.env_name = env_name
        self.agent_type = agent_type
        self.ablate_agent = ablate_agent

        self.confusion_matrix = None
        self.df = None
        self._reset()
    
    def _reset(self):
        self.df = None
        self.validities = []
        self.proximities = []
        self.sparsities = []
        self.generation_times = []

        self.domain_folders = _get_subfolders(self.test_set_path)
        self.nb_domains = len(self.domain_folders)
        self.confusion_matrix = np.zeros((self.nb_domains, self.nb_domains, self.nb_domains))
        self.examples = dict()

        if env_name.startswith("MsPacman"):
            self.pacman = True
        else:
            self.pacman = False

    def evaluate_stargan(self, generator):
        """
        Evaluates StarGAN by evaluating counterfactuals that are generated for every target action on each sample from
        the test set. A confusion matrix and a pandas data frame are returned. The confusion matrix is of dimensionality
        (nb_domains, nb_domains, nb_domains), where
        confusion_matrix[target_domain, action_on_counterfactual, original_domain] corresponds to the amount of
        counterfactuals that were originally from original_domain, generated for target_domain and are actually
        classified as action_on_counterfactual by the agent. The data frame contains information about the validity,
        proximity, sparsity and generation time.

        :param generator: The trained StarGAN generator to evaluate.
        :return: (confusion_matrix, data_frame)
        """
        def gen_fn(sample, target_domain, nb_domains):
            return generate_counterfactual(generator, sample, target_domain, nb_domains, image_size=self.img_size)
        return self._evaluate(gen_fn)

    def evaluate_olson(self, olson_agent, olson_encoder, olson_generator, olson_q, olson_p):
        """
        Evaluates Olson et al. by evaluating counterfactuals that are generated for every target action on each sample
        from the test set. A confusion matrix and a pandas data frame are returned. The confusion matrix is of
        dimensionality (nb_domains, nb_domains, nb_domains), where
        confusion_matrix[target_domain, action_on_counterfactual, original_domain] corresponds to the amount of
        counterfactuals that were originally from original_domain, generated for target_domain and are actually
        classified as action_on_counterfactual by the agent. The data frame contains information about the validity,
        proximity, sparsity and generation time.

        :param olson_agent: The agent that was used to generate the data set wrapped by src.olson.model.KerasAgent for
            the Keras-based Pac-Man agent.
        :param olson_encoder: The trained encoder.
        :param olson_generator: The trained generator.
        :param olson_q: The trained encoder Q from the Wasserstein Autoencoder.
        :param olson_p: The trained decoder P from the Wasserstein Autoencoder.
        :return: (confusion_matrix, data_frame)
        """
        def gen_fn(sample, target_domain, nb_domains):
            return generate_olson_counterfactual(sample, target_domain, olson_agent, olson_encoder, olson_generator,
                                                 olson_q, olson_p, is_pacman=self.pacman,
                                                 ablate_agent=self.ablate_agent)
        return self._evaluate(gen_fn)

    def _evaluate(self, cf_generation_fn):
        self._reset()
        total_samples = 0

        for i, domain_folder in enumerate(self.domain_folders):
            sample_file_names = os.listdir(domain_folder)
            for j, item in enumerate(sample_file_names):
                # get file name
                file_name = os.path.join(domain_folder, item)

                # load
                original = Image.open(file_name)

                for target_domain in range(self.nb_domains):
                    # generate cf
                    counterfactual, generation_time = cf_generation_fn(original, target_domain, self.nb_domains)
                    action_on_counterfactual = get_agent_action(self.agent, counterfactual, self.pacman,
                                                                agent_type=self.agent_type,
                                                                ablate_agent=self.ablate_agent)

                    # update validity, proximity, sparsity and generation time
                    np_original = np.array(original, dtype=np.float32)
                    np_counterfactual = np.array(counterfactual, dtype=np.float32)
                    self._update_confusion_matrix(i, target_domain, action_on_counterfactual)
                    self.validities.append(self.validity(target_domain, action_on_counterfactual))
                    self.proximities.append(self.proximity(np_original, np_counterfactual))
                    self.sparsities.append(self.sparsity(np_original, np_counterfactual))
                    self.generation_times.append(generation_time)
                    total_samples += 1

                    # save the first sample of every original-target-domain-mapping as an example
                    if j == 0:
                        self.examples[f"Counterfactual_{i}_Target_{target_domain}.png"] = counterfactual

                # save the first sample of every original-target-domain-mapping as an example
                if j == 0:
                    self.examples[f"Original_{i}.png"] = original
                print(f"\rFinished {int((j + 1) / len(sample_file_names) * 100)}% of domain {domain_folder}", end="")
            print()

            # print intermediate results
            print("\nIntermediate Results:")
            self._create_df()
            self.print_results()

        # calculate average values
        print("\nFinal Results:")
        self._create_df()
        self.print_results()

        return self.confusion_matrix, self.df

    def _create_df(self):
        self.df = pd.DataFrame(data={"validity": self.validities, "proximity": self.proximities,
                                     "sparsity": self.sparsities, "generation time": self.generation_times})

    def _update_confusion_matrix(self, original_domain, target_domain, action_on_counterfactual):
        self.confusion_matrix[target_domain, action_on_counterfactual, original_domain] += 1

    def _create_confusion_matrix_summary_plot(self, save_as):
        cm_summary = np.sum(self.confusion_matrix, axis=-1, dtype=int)
        action_names = get_action_names(self.env_name)[:cm_summary.shape[0]]
        for i, action_name in enumerate(action_names):
            if action_name == "LEFTFIRE":
                action_names[i] = "Left-Fire"
            elif action_name == "RIGHTFIRE":
                action_names[i] = "Right-Fire"
            elif action_name != "NOOP":
                action_names[i] = action_name.title()

        df_cm = pd.DataFrame(cm_summary, index=action_names, columns=action_names)
        plt.figure(figsize=cm_summary.shape)
        sn.heatmap(df_cm, annot=True, fmt=",")
        plt.xlabel("Action on Counterfactual")
        plt.ylabel("Target Action")
        plt.savefig(save_as, bbox_inches='tight')

    def print_results(self):
        """
        Prints the mean and standard deviation of the validity, proximity, sparsity and generation time.

        :return: None
        """
        print("MEAN RESULTS:")
        print(self.df.mean())
        print("STD RESULTS:")
        print(self.df.std(), "\n")

    def save_results(self, save_dir):
        """
        Saves the confusion matrix (as image and .npy file), the data frame (as .csv) and examples of generated
        counterfactuals.

        :param save_dir: Path to a directory that is created and filled.
        :return: None
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "Examples"))
        else:
            print("Directory ", save_dir, " already exists")
            raise FileExistsError("Directory ", save_dir, " already exists")

        np.save(os.path.join(save_dir, "Confusion_Matrix_Numpy"), self.confusion_matrix)
        self._create_confusion_matrix_summary_plot(os.path.join(save_dir, "Confusion_Matrix"))
        self.df.to_csv(os.path.join(save_dir, "Evaluation.csv"))
        for name, example in self.examples.items():
            example.save(os.path.join(save_dir, "Examples", name))

    def load_results(self, results_dir):
        """
        Loads the confusion matrix and data frame of previously saved results.

        :param results_dir: The directory that the results were saved to.
        :return: None
        """
        self.confusion_matrix = np.load(os.path.join(results_dir, "Confusion_Matrix_Numpy.npy"))
        self.df = pd.read_csv(os.path.join(results_dir, "Evaluation.csv"))

    @staticmethod
    def get_results_comparison(results_dirs):
        """
        Creates a data frame that contains a comparative summary of the given evaluation results for multiple
        approaches. This method was used to automatically generate results tables for the thesis.

        :param results_dirs: List of paths to directories with saved results.
        :return: None
        """
        columns = [("validity", ""), ("proximity", "mean"), ("proximity", "std"), ("sparsity", "mean"),
                   ("sparsity", "std"), ("generation time", "mean"), ("generation time", "std")]
        total_df = pd.DataFrame(columns=columns)
        total_df.columns = pd.MultiIndex.from_tuples(total_df.columns, names=["Metric", ""])

        for results_dir in results_dirs:
            df = pd.read_csv(os.path.join(results_dir, "Evaluation.csv"))
            data = [df["validity"].mean(), df["proximity"].mean(), df["proximity"].std(), df["sparsity"].mean(),
                    df["sparsity"].std(), df["generation time"].mean(), df["generation time"].std()]
            new_row = pd.DataFrame([data], columns=columns, index=[results_dir.split("/")[-1]])
            total_df = total_df.append(new_row)

        return total_df

    @staticmethod
    def validity(target_domain, action_on_counterfactual):
        """
        Calculates the validity of a counterfactual. The counterfactual is valid if the agent chooses the targeted
        action/domain for it.

        :param target_domain: Integer encoded target action/domain.
        :param action_on_counterfactual: The action that the agent chose on the counterfactual frame.
        :return: Bool that indicates the validity.
        """
        return target_domain == action_on_counterfactual

    @staticmethod
    def proximity(original, counterfactual):
        """
        Calculates the proximity of a counterfactual via the L1-norm normalized to range [0, 1].

        :param original: Original numpy frame.
        :param counterfactual: Counterfactual numpy frame.
        :return: The proximity between the counterfactual and the original.
        """
        return 1 - np.linalg.norm((original - counterfactual).flatten(), ord=1) / (original.size * 255)

    @staticmethod
    def sparsity(original, counterfactual):
        """
        Calculates the sparsity of a counterfactual via the L0-norm normalized to range [0, 1].

        :param original: Original numpy frame.
        :param counterfactual: Counterfactual numpy frame.
        :return: The sparsity between the counterfactual and the original.
        """
        return 1 - np.linalg.norm((original - counterfactual).flatten(), ord=0) / original.size


if __name__ == "__main__":
    restrict_tf_memory()
    GENERATE_NEW_RESULTS = False

    if GENERATE_NEW_RESULTS:
        # Settings
        ## Pacman
        pacman = True
        nb_actions = 5
        env_name = "MsPacmanNoFrameskip-v4"
        img_size = 176
        agent_file = "../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3"
        agent_type = "acer"
        model_type = "olson"
        ablate_agent = False
        agent_latent = 512 # 256 for deepq, 512 for ACER
        ## Spaceinvader
        # pacman = False
        # nb_actions = 6
        # env_name = "SpaceInvadersNoFrameskip-v4"
        # img_size = 160
        # agent_file = "../res/agents/abl_agent.tar"
        # agent_type = "olson"
        # model_type = "stargan"
        # ablate_agent = True
        # agent_latent = 32
        if agent_type == "deepq":
            agent = keras.models.load_model(agent_file)
        elif agent_type == "acer":
            agent = load_baselines_model(agent_file, num_actions=5, num_env=1)
        elif agent_type == "olson":
            # Loads a torch model with the specific architecture that Olson et al. used
            agent = olson_model.Agent(6, 32).cuda()
            agent.load_state_dict(torch.load(agent_file, map_location=lambda storage, loc: storage))
        elif agent_type == "torch_acer":
            # diry numbers for 5 actions for pacman and latent size 512
            agent = olson_model.ACER_Agent(num_actions=5, latent_size=512).cuda()
            agent.load_state_dict(torch.load(agent_file))
        elif agent_type == "torch":
            # TODO
            raise NotImplementedError("not yet implemented")

        # Create the Evaluator
        evaluator = Evaluator(agent, "res/datasets/ACER_PacMan_FearGhost2_cropped_5actions_40M_3_Unique/test", env_name,
                              img_size=img_size, agent_type=agent_type, ablate_agent=ablate_agent)

        if model_type == "stargan":
            # Load a StarGAN generator
            generator = Generator(c_dim=nb_actions, channels=3).cuda()
            generator.load_state_dict(torch.load("res/models/PacMan_FearGhost2_3/models/200000-G.ckpt",
                                                 map_location=lambda storage, loc: storage))

            # Evaluate StarGAN
            cm, df = evaluator.evaluate_stargan(generator)
            evaluator.save_results("../res/results/PacMan_FearGhost2_3")

        if model_type == "olson":
            # Load all relevant models that are necessary for the CF generation of Olson et al. via load_olson_models()
            model_name = "PacMan_FearGhost2_3_Olson"
            olson_agent, olson_encoder, olson_generator, olson_Q, olson_P = load_olson_models(
                "../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3.pt",
                "../res/models/" + model_name + "/enc39",
                "../res/models/" + model_name + "/gen39",
                "../res/models/" + model_name + "_wae/Q",
                "../res/models/" + model_name + "_wae/P",
                action_size=nb_actions,
                agent_latent=agent_latent,
                pac_man=pacman)
            # Evaluate Olson et al.
            cm_olson, df_olson = evaluator.evaluate_olson(olson_agent, olson_encoder, olson_generator, olson_Q, olson_P)
            evaluator.save_results("../res/results/" + model_name)

    # To reload old evaluation results
    else:
        pd.set_option('display.max_columns', None)
        results = Evaluator.get_results_comparison(["res/results/SpaceInvaders_Abl", "../res/results/SpaceInvaders_Abl_Olson"])
        print(results)
