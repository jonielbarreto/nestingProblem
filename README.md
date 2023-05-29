# Package nestingProblem

Python package [nestingProblem](https://github.com/jonielbarreto/nestingProblem): Optimization Algorithm for the Nesting Problem

<div style="text-align: justify"> This package can be used as a solution for the nesting problem and offers some differents approaches that can optimize the time solution. </div>

## Function

### Function that returns the packing a set of irregular-shaped
<div style="text-align: justify"> Given the set of images of 2D objects and some features of the problem, this function solves the nesting problem of the images in a pre-established specific size. The result can be shown as images and a print of the execution time. </div>

```markdown

nestin_probl_funct(PATH,
		   page_size = page_r,
		   stride = step,
		   function_name = 'pac.FIRST_START')
```

#### Parameters
* **PATH**: _list_<br/>
> Expected a path of all objects, each in a different image and images that OpenCV can read.

* **page_size**: _list of int [page_height, page_width]_<br/>
> Size of the page (space) where the objects must be packing. Expected a list of two elements. Default value: Paper A4 size. 

* **stride**: _int_<br/>
> Step size for choosing contour points. Default value: 3.

* **function_name**: _str_<br/>
> Choice of approach to solving the nesting problem. Expected one of ('pac.FIRST_START', 'pac.WORST_FIRST', 'pac.BEST_START', 'pac.RANDOM_FIRST'). Default value: 'pac.FIRST_START'

#### _Returns_
* **list_ofPages**: _list_<br/>
> List of pages used to package the objects.

It also prints the execution time. 
