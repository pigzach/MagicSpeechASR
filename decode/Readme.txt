### Dir
- run.gpu.sh: run with gpu, the the nj equals to speaker number except ivector
- run.cpu.sh: run with cpu, and the nj equals to speaker number except ivector

### Check
- ls /home  # check the home dir
- !unzip -d /home/jovyan/work/challenge_final challenge_final.zip
- !locale -a
- How to Install en_US.utf-8
  - !apt-get -y install locales
  - !locale-gen en_US.utf-8

### Run
- mv /home/jovyan/work/challenge_final /opt/kaldi/egs/challenge_final
- cd /opt/kaldi/egs/challenge_final
- !bash clean.sh
- !chmod 755 run.cpu.sh
- !bash run.cpu.sh /home/jovyan/data/[corpus-dir]
- cp -f submission.csv /home/jovyan/work/submission.csv

### Final
- !du -sh
