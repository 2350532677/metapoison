# metapoison anonymous submission


# Installation instructions

Install all packages as listed in ```environment.yml```, for example via anaconda.

# Running instructions:
* If you have a comet account, fill in your information in the ```.comet.config file```.

* Use ```train.py``` to generate weight samples and upload them to your comet API.

* Call ```main.py``` with your options (various options are defined in the argparser) to craft poisons.

* Run ```victim.py``` to validate the crafted poisons on a new network trained from scratch.

