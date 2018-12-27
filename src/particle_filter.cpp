/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cassert>
#include <set>

#include "particle_filter.h"

using namespace std;

namespace {
static constexpr float MAX_DIST = 99999999;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 200;  // TODO: Set the number of particles

  particles.resize(num_particles);

  default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    particles[i].id     = i;
    particles[i].x      = dist_x(gen);
    particles[i].y      = dist_y(gen);
    particles[i].theta  = dist_theta(gen);
    particles[i].weight = 1;
  }

  // weight/associations to be updated after measurement
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x

  if (fabs(yaw_rate) < 0.0001)
  {
    yaw_rate = 0.0001;
  }

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[0]);
  normal_distribution<double> dist_theta(0, std_pos[0]);

  for (auto & particle : particles)
  {
    particle.x     = particle.x + cos(particle.theta + yaw_rate * delta_t) * velocity * delta_t;
    particle.y     = particle.y + sin(particle.theta + yaw_rate * delta_t) * velocity * delta_t;
    particle.theta = particle.theta + yaw_rate * delta_t;

    particle.x     = particle.x + dist_x(gen);
    particle.y     = particle.y + dist_y(gen);
    particle.theta = particle.theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  for (auto & observation : observations)
  {
    // Reset observation landmark id to -1 before searching
    observation.id = -1;
    float min_dist = MAX_DIST;
    int selected = -1;

    for (size_t i = 0; i < predicted.size(); ++i)
    {
      // If this landmark has already been used by one observation
      if (predicted[i].id == -1)
      {
        continue;
      }

      auto x = predicted[i].x - observation.x;
      auto y = predicted[i].y - observation.y;
      auto dist = sqrt(x * x + y * y);

      if (dist < min_dist)
      {
        min_dist = dist;
        selected = i;
        // Mark predicted id as already used
      }
    }

    if (selected != -1)
    {
      observation.id = predicted[selected].id;
      predicted[selected].id = -1;
    }

    //cout << "x = " << observation.x << " y = " << observation.y
    //     << " land.x = " << predicted[selected].x << " land.y = "
    //     << predicted[selected].y << " min_dist = " << min_dist << endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  // The list of all landmarks
  if (observations.empty())
  {
    cout << "No observation. Ignore weights update" << endl;
    return;
  }

  std::vector<LandmarkObs> predicted;
  for (auto & landmark : map_landmarks.landmark_list)
  {
    predicted.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
  }

  int i = 0, j = 0;
  for (auto & particle : particles)
  {
    ++i;
    // First convert observations to map frame reference
    // Make a copy of the original observation
    std::vector<LandmarkObs> observations_map = observations;
    for (auto & observation : observations_map)
    {
      LandmarkObs obs = observation;
      obs.x = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
      obs.y = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
      observation = obs;
    }

    // Second find associations of each particle/measurement
    dataAssociation(predicted, observations_map);
    // Now each of observations_map will point to one landmark

    vector<int> associations;
    vector<double> sense_x, sense_y;

    particle.weight = 1;
    j = 0;
    for (auto & observation: observations_map)
    {
      ++j;
      // This should not happen
      assert(observation.id != -1);

      // Third update weights if observation id is not -1
      float x  = predicted[observation.id - 1].x - observation.x;
      float x2 = x * x;
      float y  = predicted[observation.id - 1].y - observation.y;
      float y2 = y * y;

      float exponent = exp(-(x2 / (2 * std_landmark[0] * std_landmark[0]) + y2 / (2 * std_landmark[1] * std_landmark[1])));
      particle.weight *= 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]) * exponent;
      //cout << "exponent = " << particle.weight << endl;

      associations.push_back(observation.id);
      sense_x.push_back(observation.x);
      sense_y.push_back(observation.y);
    }

    SetAssociations(particle, associations, sense_x, sense_y);
  }
  // Finally normalise the weights for resampling
  float weight_sum = 0;
  for (auto & particle : particles)
  {
    weight_sum += particle.weight;
  }
  for (auto & particle : particles)
  {
    particle.weight /= weight_sum;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<float> weights;
  for (auto & particle : particles)
  {
    weights.push_back(particle.weight);
  }

  default_random_engine gen;
  discrete_distribution<> d(begin(weights), end(weights));

  auto resampled = vector<Particle>();
  num_particles = particles.size();

  for (int i = 0; i < num_particles; ++i)
  {
    auto index = d(gen);
    resampled.push_back(particles[index]);
  }

  particles = resampled;

#if 0
  // Another implementation Spinning wheel
  float beta = 0;
  auto w_max = max(w);
  size_t index = int(random.random() * num_samples);
  for (size_t i = 0; i < num_samples; ++i)
  {
    beta = beta + random.random() * 2 * w_max;
    while (beta > w[index])
    {
        beta = beta - w[index]
        index = (index + 1) % num_samples;
    }
    resamples.push_back(particles[index]);
  }
#endif
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
