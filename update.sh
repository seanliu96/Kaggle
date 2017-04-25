echo "##Author: Sean Liu" > README.md
date | xargs -I{} echo "##TIME:" {} >> README.md
echo "
Kaggle
" >> README.md

git add `git ls-files -o`
git add `git ls-files -m`
git rm `git ls-files -d`
date | xargs -I{} git commit -m {}
git push origin master
