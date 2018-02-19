import numpy as np
import requests
import getopt
import fileinput
import sys
import scipy 
import math

data_file = open(sys.argv[1], 'r')
shuffled_data_file = open(sys.argv[2], 'r')
shuffled_data_file1 = open(sys.argv[2], 'r')
iris_data = np.loadtxt(data_file,dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'label'),'formats': (np.float, np.float, np.float, np.float, '|S15')},delimiter=',', skiprows=0)
iris_shuffled_data = np.genfromtxt(shuffled_data_file, delimiter=',', usecols=(0,1,2,3))
iris_shuffled_data1 = np.loadtxt(shuffled_data_file1,dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'label'),'formats': (np.float, np.float, np.float, np.float, '|S15')},delimiter=',', skiprows=0)

i = 2
# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows,column,value):
   # Make a function that tells us if a row is in the first group (true) or the second group (false)
   split_function=None
   if isinstance(value,int) or isinstance(value,float): # check if the value is a number i.e int or float
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value
   
   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)
def uniquecounts(rows):
   results={}
   for row in rows:
      # The result is the last column
      r=row[len(row)-1]
      if r not in results: results[r]=0
      results[r]+=1
   return results
# Entropy is the sum of p(x)log(p(x)) across all 
# the different possible results
def entropy(rows):
   from math import log
   log2=lambda x:log(x)/log(2)  
   results=uniquecounts(rows)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent
class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb
def buildtree(rows,scoref=entropy): #rows is the set, either whole dataset or part of it in the recursive call, 
                                    #scoref is the method to measure heterogeneity. By default it's entropy.
  if len(rows)==0: return decisionnode() #len(rows) is the number of units in a set
  current_score=scoref(rows)

  # Set up some variables to track the best criteria
  best_gain=0.0
  best_criteria=None
  best_sets=None
  
  column_count=len(rows[0])-1   #count the # of attributes/columns. 
                                #It's -1 because the last one is the target attribute and it does not count.
  for col in range(0,column_count):
    # Generate the list of all possible different values in the considered column
    global column_values        #Added for debugging
    column_values={}            
    for row in rows:
       column_values[row[col]]=1   
    # Now try dividing the rows up for each value in this column
    for value in column_values.keys(): #the 'values' here are the keys of the dictionnary
      (set1,set2)=divideset(rows,col,value) #define set1 and set2 as the 2 children set of a division
      
      # Information gain
      p=float(len(set1))/len(rows) #p is the size of a child set relative to its parent
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) #cf. formula information gain
      if gain>best_gain and len(set1)>0 and len(set2)>0: #set must not be empty
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)
        
  # Create the sub branches   
  if best_gain>0:
    trueBranch=buildtree(best_sets[0])
    falseBranch=buildtree(best_sets[1])
    return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    return decisionnode(results=uniquecounts(rows))
tree=buildtree(iris_data)
def printtree(tree,indent=''):
   # Is this a leaf node?
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.value)+'? ')
        # Print the branches
        print indent+'True->',
        printtree(tree.tb,indent+'  ')
        print indent+'False->',
        printtree(tree.fb,indent+'  ')


def getwidth(tree):
  if tree.tb==None and tree.fb==None: return 1
  return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
  if tree.tb==None and tree.fb==None: return 0
  return max(getdepth(tree.tb),getdepth(tree.fb))+1


from PIL import Image,ImageDraw

def drawtree(tree,jpeg='tree.jpg'):
  w=getwidth(tree)*100
  h=getdepth(tree)*100+120

  img=Image.new('RGB',(w,h),(255,255,255))
  draw=ImageDraw.Draw(img)

  drawnode(draw,tree,w/2,20)
  img.save(jpeg,'JPEG')

def drawnode(draw,tree,x,y):
  global i
  if tree.results==None:
		# Get the width of each branch
		w1=getwidth(tree.fb)*100
		w2=getwidth(tree.tb)*100
		# Determine the total space required by this node
		left=x-(w1+w2)/2
		right=x+(w1+w2)/2
		# Draw the condition string
		draw.text((x-20,y-10),str(tree.value),(0,0,0))
		# Draw links to the branches
		draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
		draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))
		with open("Q1.txt", "a") as text_file:
			text_file.write("\n{0} ".format(str(tree.value)))
			text_file.write("{0} ".format(i))
			i = i + 1
			text_file.write("{0}".format(i))
			i = i + 1
		# Draw the branch nodes
		drawnode(draw,tree.fb,left+w1/2,y+100)
		drawnode(draw,tree.tb,right-w2/2,y+100)
  else:
		txt=' \n'.join(['%s:\n%d'%v for v in tree.results.items()])
		draw.text((x-20,y),txt,(0,0,0))
		with open("Q1.txt", "a") as text_file:
			text_file.write("{0}".format(str(tree.results)))

def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
  return classify(observation,branch)

with open("Q1.txt", "w") as text_file:
  text_file.write( "\n\nAnswer for Q1:\n\n")
drawtree(tree,jpeg='treeview.jpg')	  
  
with open("Q2.txt", "w") as text_file:
	text_file.write( "\n\nAnswer for Q2:\n\n")
for line, line1 in zip(iris_shuffled_data, iris_shuffled_data1):
	with open("Q2.txt", "a") as text_file:
		text_file.write("Predicted Label:  {0};  ".format(str(line1[4])))
		text_file.write("True Label:  {0}\n".format(classify(line,tree)))
print "\nThe tree is as below:\n"
printtree(tree)
print "\nEntropy value: "
print(entropy(iris_data))
print "\nUnique data:"
print(uniquecounts(iris_data))