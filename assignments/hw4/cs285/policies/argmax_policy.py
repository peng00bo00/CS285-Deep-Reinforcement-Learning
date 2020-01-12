# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class ArgMaxPolicy(object):

    def __init__(self, sess, critic):
        self.sess = sess
        self.critic = critic

        # TODO: Define what action this policy should return
        # HINT1: the critic's q_t_values indicate the goodness of observations, 
        # so they should be used to decide the action to perform
        self.action = tf.argmax(self.critic.q_t_values, axis=1)

    def get_action(self, obs):

        # TODO: Run the tensor defined above
        # HINT1: you'll want to populate placeholders that are required for the critic to calculate its q_t_values
        # HINT2: see the obs_t_ph placeholder 
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        return self.sess.run([self.action], feed_dict={self.critic.obs_t_ph: observation})[0]
