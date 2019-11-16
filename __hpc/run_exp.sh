rm *.sh.e*
rm *.sh.o*

for file in $1/*
do
  qsub "$file"
done