# dreaming-hawk

A testing bed

## Todo:

-   Work on paragraph semantic optimization
-   CLI Tool QOL features
    -   Start with from end word deletion.
-   [Create a Javascript frontend to visualize the graph]
-   Implementation of Delete Word
    -   Decrement Node DONE
    -   Remove temporal edges
    -   Remove from the sentence window
    -   Remove from the paragraph window
    -   Remove from actual window

## Design for live text imputation

Javascript Web Application -> Python Wrapper Class for Text Imputation -> WordGraph with LemmaGraph.

Javascript web application should be a text editor to use and sends the texts to the Python wrapper.

Python wrapper should make commands to the WordGraph. There should be some preprocessing done, by assembling a stack so that we can minimize function calls and do things in batches.

Keep track of the sentence window in the graph.
<video src="./ChalmersTextDemo.mp4" controls width="640"></video>
