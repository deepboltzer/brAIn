# Improvements

## <s>Re-run the `_steps.ipynb` files</s>

<s>Re-run the `_steps.ipynb` files; this time increase the number of evaluation episodes to 1000 timesteps. Hopefully that will reduce the chance that the agent just performs luckily in the early stages.

That way plot would be more pleasant to look at and also more intuitive to explain.</s>

Tried it; didn't matter, but took ages to run.

Maybe instead try to save the model more frequently.
Remember to re-run the evaluation and visualisation too and modify the README. <mark>done</mark>

## Uniform attack

- Update README with uniform attack
- <s>Uniform attack track amount of pertubation</s>
- <s>Consider calculated target action in pertubation of observation</s>
- Maybe change how mean perturbation is calculated; right now:
  - adversary doesn't interfere much
  - agent performs well
  - episodes are long
  - perturbation sums up to a high value
- <s>move perturb() into separate python file</s>
- rerun and remove division by timestep for mean perturbation
  - also do something about the difference in the threshold for x and theta

## Strategically timed attack

- <s>results do not seem right</s>

## Update README with info on stable-baselines3

## Move learn+save code in utility file

## Maybe change file names

## Clean-up over-all structure

## Create function get_save_dir() in utils.py
