# Git Workfow  

This file describes our workflow with git and several branches. Additional commands are added for support of the project.

### Working on Branches

- `git branch` : Lists all branches
- `git branch [branch_name]`: Creates new branch with given name  
- `git checkout [branch_name]`: Switches to branch
- `git branch [branch_name] --delete`: Deletes Branch with given name, works only if branch was changed before i.e. to main.

If you are working on a created branch, then you can apply all git commands like `git add, git commit ...` as usual to push your changes to your 
local branch.
To perform a push to your branch, this cmd is probably needed to set a new upstream:
`git push --set-upstream origin [branch_anem]`

### Merge branch into Master

To merge a  branch into master/main then you need to create a pull request from your
branch to the master/main branch. This can be done online on github.com. If you make a new *pull request* you should see your pushed changes to your branch and add a description of what you particularly have changed. After you created a request, somebody of the team has to review the changes and approve it.



