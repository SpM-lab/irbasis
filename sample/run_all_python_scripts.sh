for file in `ls *.py`
do
  python $file
  # exit if it fails
  [ $? -eq 0 ] || exit $?;
done
