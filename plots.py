def test(self, start_node):
    # reset agent/environment state
    self.distance_travelled = 0.
    self.total_reward = 0.
    self.current_state = [self.initial_node, self.node_combinations.index(set([]))]
    route = [self.current_state[0]]

    for t in itertools.count():
        # select action using policy
        actions = self.listPossibleActions(board)
        selected_action = self.greedyPolicy(actions=actions)

        # Update distance travelled
        self.distance_travelled += self.graph.returnVertexWeight(node1=self.current_state[0],
                                                                 node2=selected_action)
        # Update state
        self.current_state = self.transitionState(state=self.current_state, action=selected_action)

        # Terminate if new state
        if self.isStateTerminal(state=self.current_state):
            convergence = True

            # rotate best route
            while route[0] != start_node:
                route.append(route.pop(0))
            route.append(start_node)
            return route, convergence

        elif t > 10000:
            print('NO CONVERGENCE')
            convergence = False
            return route, convergence

        else:
            route.append(self.current_state[0])


def plotRewards(self, save_name=None):
    plt.figure()
    plt.xlabel('$Episodes$')
    plt.ylabel('$Total \; Reward$')
    plt.ylim()
    plt.xlim(0, len(self.analytics['episode_rewards']))
    plt.plot(self.analytics['episode_rewards'], 'o-', markersize=0)
    plt.xticks([i for i in
                np.arange(0, len(self.analytics['episode_rewards']), len(self.analytics['episode_rewards']) / 10)])
    plt.grid(linestyle='--', which='both', axis='both')
    if save_name:
        if not os.path.exists('./plots/'):
            os.mkdir('./plots/')
        plt.savefig(fname='./plots/' + save_name + '.png', dpi=800)
    plt.show()


def plotDistance(self, save_name=None):
    plt.figure()
    plt.xlabel('$Episodes$')
    plt.ylabel('$Total \; Distance \; Travelled$')
    plt.ylim()
    plt.xlim(0, len(self.analytics['episode_distances']))
    plt.plot(self.analytics['episode_distances'], 'o-', markersize=0)
    plt.xticks([i for i in np.arange(0, len(self.analytics['episode_distances']),
                                     len(self.analytics['episode_distances']) / 10)])
    plt.grid(linestyle='--',
             which='both',
             axis='both')
    if save_name:
        if not os.path.exists('./plots/'):
            os.mkdir('./plots/')
        plt.savefig(fname='./plots/' + save_name + '.png', dpi=800)
    plt.show()


def plotEpisodeLength(self, save_name=None):
    plt.figure()
    plt.xlabel('$Episodes$')
    plt.ylabel('$Episode \; Length$')
    plt.ylim()
    plt.xlim(0, len(self.analytics['episode_lengths']))
    plt.plot(self.analytics['episode_lengths'], 'o-', markersize=0, zorder=1)
    plt.xticks([i for i in np.arange(0, len(self.analytics['episode_lengths']),
                                     len(self.analytics['episode_lengths']) / 10)])
    plt.grid(linestyle='--',
             which='both',
             axis='both',
             zorder=3)
    plt.hlines(y=len(self.graph.list_nodes),
               xmin=0,
               xmax=len(self.analytics['episode_lengths']),
               colors='r',
               linestyles='dashed',
               zorder=2)
    if save_name:
        if not os.path.exists('./plots/'):
            os.mkdir('./plots/')
        plt.savefig(fname='./plots/' + save_name + '.png', dpi=800)
    plt.show()


def plotEpsilon(self, save_name=None):
    plt.figure()
    plt.xlabel('$Episodes$')
    plt.ylabel('$Epsilon$')
    plt.ylim()
    plt.xlim(0, len(self.analytics['epsilon_values']))
    plt.plot(self.analytics['epsilon_values'], 'o-', markersize=0)
    plt.xticks([i for i in np.arange(0, len(self.analytics['epsilon_values']),
                                     len(self.analytics['epsilon_values']) / 10)])
    plt.grid(linestyle='--', which='both', axis='both')
    if save_name:
        if not os.path.exists('./plots/'):
            os.mkdir('./plots/')
        plt.savefig(fname='./plots/' + save_name + '.png', dpi=800)
    plt.show()


    def epsilonGreedyPolicy(self, actions, decay_rate, decay):
        """
        #TODO: put epsilon as argument of function, then return new epsilon?
        :param decay_rate:
        :param actions:
        :param decay:
        :return:
        """

        if random.uniform(0, 1) <= self.epsilon:
            random.shuffle(actions)
            selected_action = actions[0]
        else:
            # create list of actions with the largest Q value
            max_Q = self.Q_table[tuple(self.current_state)][actions[0]]
            list_max_Q_actions = [actions[0]]
            for i in range(1, len(actions)):
                if self.Q_table[tuple(self.current_state)][actions[i]] > max_Q:
                    max_Q = self.Q_table[tuple(self.current_state)][actions[i]]
                    list_max_Q_actions = [actions[i]]
                elif self.Q_table[tuple(self.current_state)][actions[i]] == max_Q:
                    list_max_Q_actions.append(actions[i])
                else:
                    continue
            # randomly select one of the actions with the largest Q value
            random.shuffle(list_max_Q_actions)
            selected_action = list_max_Q_actions[0]

        # Agent more likely to be greedy after each iteration
        if decay:
            if self.epsilon >= 0.5:
                self.epsilon = self.epsilon * (1. - decay_rate)
            else:
                self.epsilon = self.epsilon * (1. - decay_rate / 10.)

        return selected_action