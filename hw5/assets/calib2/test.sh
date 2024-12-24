i=1
for file in *.bmp; do
    mv "$file" "img$(printf "%02d" $i).bmp"
    i=$((i+1))
done