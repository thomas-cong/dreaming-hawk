# dreaming-hawk

A testing bed

## Todo:

-   Optimize semantic enrichment to work live
    -   In progress
    -   NEED: work on a text workhorse function. the function should take text- and unilaterally extract word, sentence, and other info. it should return when sentences end, along with those words that were given. can be modified to return lemmas, pos, embeddings, etc., but this shouldn't be default. default should give the words, sentences, paragraphs, and where they end to be used for live embedding. DONE
    -   Semantic enrichment at a sentence level DONE
    -   Write tests for previous two features. TODO
-   [Add a way to export the graph to a file (probably JSON format)]
    -   DONE
-   [Create a Javascript frontend to visualize the graph]
-   Implementation of Delete Word
    -   Decrement Node
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
