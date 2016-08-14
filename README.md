Captcha Neural Guesser
======================

```bash
sudo apt-get install imagemagick++-dev libtbb-dev
git submodule init && git submodule update
cmake -DUSE_TBB=ON .
./learner 2000 1000 20
```
