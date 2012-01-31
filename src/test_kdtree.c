/** @file   test_kdtree.c
 ** @author Junji Torii
 ** @breif  Test vl/kdtree.h
 **/

#include <stdio.h>
#include <assert.h>

#include <vl/random.h>
#include <vl/kdtree.h>

VlKDForest *
kdtreebuild(int verbose,
            vl_size dimension,
            vl_size numTrees,
            int thresholdingMethod,
            vl_size numData, float *data)
{
    vl_type dataType = VL_TYPE_FLOAT;
    VlKDForest * forest;

    forest = vl_kdforest_new (dataType, dimension, numTrees) ;
    vl_kdforest_set_thresholding_method (forest, thresholdingMethod) ;

    if (verbose) {
        char const * str = 0 ;
        printf("vl_kdforestbuild: data %s [%d x %d]\n",
                  vl_get_type_name (dataType), dimension, numData) ;
        switch (vl_kdforest_get_thresholding_method(forest)) {
        case VL_KDTREE_MEAN : str = "mean" ; break ;
        case VL_KDTREE_MEDIAN : str = "median" ; break ;
        default: abort() ;
        }
        printf("vl_kdforestbuild: threshold selection method: %s\n", str) ;
        printf("vl_kdforestbuild: number of trees: %d\n",
               vl_kdforest_get_num_trees(forest)) ;
    }

    /* -----------------------------------------------------------------
     *                                                            Do job
     * -------------------------------------------------------------- */

    vl_kdforest_build (forest, numData, data) ;

    if (verbose) {
        vl_uindex ti ;
        for (ti = 0 ; ti < vl_kdforest_get_num_trees(forest) ; ++ ti) {
            printf("vl_kdforestbuild: tree %d: depth %d, num nodes %d\n",
                   ti,
                   vl_kdforest_get_depth_of_tree(forest, ti),
                   vl_kdforest_get_num_nodes_of_tree(forest, ti)) ;
        }
    }
    
    return(forest);
}

int
save_data(const char *fname, float *data, vl_size dim, vl_size num)
{
    FILE *fp;
    size_t n = 0;
    int i;

    if((fp = fopen(fname, "wb")) == NULL){
        printf("can't open file\n");
        return -1;
    }
    fwrite(&dim, sizeof(vl_size), 1, fp);
    fwrite(&num, sizeof(vl_size), 1, fp);
    for(i = 0;i < num; i++){
        n += fwrite(&data[dim * i], sizeof(float), dim, fp);
    }
    fclose(fp);

    return n;
}

float *
load_data(const char *fname, vl_size *dim, vl_size *num)
{
    FILE *fp;
    int i;
    float *data;

    if((fp = fopen(fname, "rb")) == NULL){
        printf("can't open file\n");
        return -1;
    }    

    fread(dim, sizeof(vl_size), 1, fp);
    fread(num, sizeof(vl_size), 1, fp);

    data = (float *)malloc((*dim) * (*num) * sizeof(float));
    for(i = 0;i < *num; i++){
        fread(&data[*dim * i], sizeof(float), *dim, fp);
    }
    
    return data;
}

float *
create_data(int dim, int num)
{
    float *data;
    int i, j;
    vl_uint32 init [4] = {0x123, 0x234, 0x345, 0x456} ;
    VlRand rand ;
    vl_rand_init (&rand) ;

    vl_rand_seed_by_array (&rand, init, sizeof(init)/sizeof(init[0])) ;

    data = (float *)malloc(dim * num * sizeof(float));
    if(data == NULL) return NULL;

    for(i = 0;i < num; i++){
        for(j = 0;j < dim; j++){
            data[i * dim + j] = (float)vl_rand_real2(&rand);
        }
    }
    return data;
}

float
dist_l2(int dim, float *d1, float *d2)
{
    int i;
    double sum = 0;
    for(i = 0;i < dim; i++){
        sum += (d1[i] - d2[i]) * (d1[i] - d2[i]);
    }
    return (float)sum;
}

size_t
read_VlKDTreeNode(FILE *fp, VlKDTreeNode *node)
{
    size_t n = 0;

    n += fread(&(node->parent), sizeof(vl_uindex), 1, fp);
    n += fread(&(node->lowerChild), sizeof(vl_index), 1, fp);
    n += fread(&(node->upperChild), sizeof(vl_index), 1, fp);
    n += fread(&(node->splitDimension), sizeof(unsigned int), 1, fp);
    n += fread(&(node->splitThreshold), sizeof(double), 1, fp);
    n += fread(&(node->lowerBound), sizeof(double), 1, fp);
    n += fread(&(node->upperBound), sizeof(double), 1, fp);

    return n;
}

size_t
write_VlKDTreeNode(FILE *fp, VlKDTreeNode *node)
{
    size_t n = 0;

    n += fwrite(&(node->parent), sizeof(vl_uindex), 1, fp);
    n += fwrite(&(node->lowerChild), sizeof(vl_index), 1, fp);
    n += fwrite(&(node->upperChild), sizeof(vl_index), 1, fp);
    n += fwrite(&(node->splitDimension), sizeof(unsigned int), 1, fp);
    n += fwrite(&(node->splitThreshold), sizeof(double), 1, fp);
    n += fwrite(&(node->lowerBound), sizeof(double), 1, fp);
    n += fwrite(&(node->upperBound), sizeof(double), 1, fp);

    return n;
}

size_t
read_VlKDTreeDataIndexEntry(FILE *fp, VlKDTreeDataIndexEntry *index)
{
    size_t n = 0;

    n += fread(&(index->index), sizeof(vl_index), 1, fp);
    n += fread(&(index->value), sizeof(double), 1, fp);

    return n;
}

size_t
write_VlKDTreeDataIndexEntry(FILE *fp, VlKDTreeDataIndexEntry *index)
{
    size_t n = 0;

    n += fwrite(&(index->index), sizeof(vl_index), 1, fp);
    n += fwrite(&(index->value), sizeof(double), 1, fp);

    return n;
}

size_t
read_VlKDTree(FILE *fp, VlKDForest * self, VlKDTree *tree)
{
    size_t n = 0;
    vl_size i;

    n += fread(&(tree->numUsedNodes), sizeof(vl_size), 1, fp);
    n += fread(&(tree->numAllocatedNodes), sizeof(vl_size), 1, fp);
    n += fread(&(tree->depth), sizeof(unsigned int), 1, fp);

    tree->nodes = vl_malloc(sizeof(VlKDTreeNode) * tree->numAllocatedNodes);
    for(i = 0;i < tree->numAllocatedNodes; i++){
        n += read_VlKDTreeNode(fp, &tree->nodes[i]);
    }

    tree->dataIndex = vl_malloc(sizeof(VlKDTreeDataIndexEntry) * self->numData);
    for(i = 0;i < self->numData ; i++){
        n += read_VlKDTreeDataIndexEntry(fp, &tree->dataIndex[i]);
    }

    return n;
}

size_t
write_VlKDTree(FILE *fp, VlKDForest * self, VlKDTree *tree)
{
    size_t n = 0;
    vl_size i;

    n += fwrite(&(tree->numUsedNodes), sizeof(vl_size), 1, fp);
    n += fwrite(&(tree->numAllocatedNodes), sizeof(vl_size), 1, fp);
    n += fwrite(&(tree->depth), sizeof(unsigned int), 1, fp);
    

    for(i = 0;i < tree->numAllocatedNodes; i++){
        n += write_VlKDTreeNode(fp, &tree->nodes[i]);
    }

    for(i = 0;i < self->numData; i++){
        n += write_VlKDTreeDataIndexEntry(fp, &tree->dataIndex[i]);
    }
    
    return n;
}

size_t
read_VlKDForest(FILE *fp, VlKDForest * self)
{
    size_t n = 0;
    vl_uindex ti;

    n += fread(&(self->dimension), sizeof(vl_size), 1, fp);
    n += fread(&(self->dataType), sizeof(vl_type), 1, fp);
    n += fread(&(self->numData), sizeof(vl_size), 1, fp);
    n += fread(&(self->numTrees), sizeof(vl_size), 1, fp);
    n += fread(&(self->thresholdingMethod), sizeof(VlKDTreeThresholdingMethod), 1, fp);
    n += fread(&(self->splitHeapNumNodes), sizeof(vl_size), 1, fp);
    n += fread(&(self->splitHeapSize), sizeof(vl_size), 1, fp);

    n += fread(&(self->searchHeapNumNodes), sizeof(vl_size), 1, fp);
    n += fread(&(self->searchId), sizeof(vl_uindex), 1, fp);
    n += fread(&(self->searchMaxNumComparisons), sizeof(vl_size), 1, fp);

    self->trees = vl_malloc (sizeof(VlKDTree*) * self->numTrees);
    for (ti = 0 ; ti < self->numTrees ; ++ ti) {
        self->trees[ti] = vl_malloc (sizeof(VlKDTree));
        n += read_VlKDTree(fp, self, self->trees[ti]);
    }

    return n;
}

size_t
write_VlKDForest(FILE *fp, VlKDForest * self)
{
    size_t n = 0;
    vl_uindex ti;

    n += fwrite(&(self->dimension), sizeof(vl_size), 1, fp);
    n += fwrite(&(self->dataType), sizeof(vl_type), 1, fp);
    n += fwrite(&(self->numData), sizeof(vl_size), 1, fp);
    n += fwrite(&(self->numTrees), sizeof(vl_size), 1, fp);
    n += fwrite(&(self->thresholdingMethod), sizeof(VlKDTreeThresholdingMethod), 1, fp);
    n += fwrite(&(self->splitHeapNumNodes), sizeof(vl_size), 1, fp);
    n += fwrite(&(self->splitHeapSize), sizeof(vl_size), 1, fp);

    n += fwrite(&(self->searchHeapNumNodes), sizeof(vl_size), 1, fp);
    n += fwrite(&(self->searchId), sizeof(vl_uindex), 1, fp);
    n += fwrite(&(self->searchMaxNumComparisons), sizeof(vl_size), 1, fp);

    for (ti = 0 ; ti < self->numTrees ; ++ ti) {
        n += write_VlKDTree(fp, self, self->trees[ti]);
    }
    
    return n;
}

VlKDForest *
load_VlKDForest(const char *fname)
{
    FILE *fp;
    size_t n;
    VlKDForest *self = vl_malloc (sizeof(VlKDForest)) ;

    if((fp = fopen(fname, "rb")) == NULL)
        return -1;

    n = read_VlKDForest(fp, self);
    fclose(fp);

    self -> rand = vl_get_rand ();
    self -> searchHeapArray = 0;
    self -> searchIdBook = 0;

    switch (self->dataType) {
    case VL_TYPE_FLOAT:
        self -> distanceFunction = (void(*)(void))
            vl_get_vector_comparison_function_f (VlDistanceL2) ;
        break ;
    case VL_TYPE_DOUBLE :
        self -> distanceFunction = (void(*)(void))
            vl_get_vector_comparison_function_d (VlDistanceL2) ;
        break ;
    default :
        abort() ;
    }

    return self;
}

size_t
save_VlKDForest(const char *fname, VlKDForest * self)
{
    FILE *fp;
    size_t n;

    if((fp = fopen(fname, "wb")) == NULL)
        return -1;

    n = write_VlKDForest(fp, self);
    fclose(fp);

    return n;
}

void
test_simple(int verbose)
{
    VlKDForest *forest;
    int i, j;
    float *data, *query;
    vl_size dim = 128;
    vl_size num = 10000;
    vl_size numTrees = 1;

    /*
     * create a test data
     */
    if((data = create_data(dim ,num)) == NULL){
        printf("not enough memoey\n");
        exit(1);
    }
    if(verbose) printf("has created a test data\n");

    if((query = (float *)malloc(dim * sizeof(float))) == NULL){
        printf("not enough memoey\n");
        exit(1);
    }
    for(i = 0;i < dim; i++) query[i] = 0.5;
    if(verbose) printf("has created a query\n");

    /*
     * build a kd-tree forest
     */
    forest = kdtreebuild(1, dim, numTrees, VL_KDTREE_MEDIAN, num, data);
    if(verbose) printf("has created a forest\n");
    
    if(verbose && 0){
        for(j = 0;j < numTrees; j++){
            printf("dataIndex[%d] = [", j);
            for(i = 0;i < forest->numData; i++){
                printf("%d ",
                       forest->trees[j]->dataIndex[i].index + 1);
            }
            printf("]\n");
        }
    }

    /*
     * save
     */
    save_data("data.bin", data, dim, num);
    save_VlKDForest("forest.bin", forest);

    /*
     * search neighbors
     */
    vl_size numNeighbors = 10;
    unsigned int numComparisons = 0 ;
    unsigned int maxNumComparisons = 0 ;
    VlKDForestNeighbor * neighbors ;

    vl_kdforest_set_max_num_comparisons (forest, maxNumComparisons) ;
    neighbors = vl_malloc (sizeof(VlKDForestNeighbor) * numNeighbors) ;

    numComparisons = vl_kdforest_query (forest, neighbors, numNeighbors,
                                        query);

    for(i = 0;i < numNeighbors; i++){
        printf("%d %f\n",
               neighbors[i].index + 1, neighbors[i].distance);

        /* check distance */
        if(fabs(
                dist_l2(dim, query, &data[neighbors[i].index * dim]) -
                neighbors[i].distance) > 1e-6){
            printf("%d distance is different. %f\n",
                  dist_l2(dim, query, &data[neighbors[i].index * dim]) );
        }
        /* check order */
        if(i != 0 && neighbors[i-1].distance > neighbors[i].distance){
            printf("order is wrong.\n");
        }
    }

    vl_free(neighbors);
    vl_kdforest_delete(forest);
    free(data);
    free(query);
}

void
test_query_from_file(int verbose)
{
    VlKDForest *forest;
    int i;
    float *data, *query;
    vl_size dim;
    vl_size num;

    /*
     * load
     */
    data = load_data("data.bin", &dim, &num);
    forest = load_VlKDForest("forest.bin");
    forest->data = data;

    if((query = (float *)malloc(dim * sizeof(float))) == NULL){
        printf("not enough memoey\n");
        exit(1);
    }
    for(i = 0;i < dim; i++) query[i] = 0.5;
    if(verbose) printf("has created a query\n");

    /*
     * search neighbors
     */
    vl_size numNeighbors = 10;
    unsigned int numComparisons = 0 ;
    unsigned int maxNumComparisons = 0 ;
    VlKDForestNeighbor * neighbors ;

    vl_kdforest_set_max_num_comparisons (forest, maxNumComparisons) ;
    neighbors = vl_malloc (sizeof(VlKDForestNeighbor) * numNeighbors) ;

    numComparisons = vl_kdforest_query (forest, neighbors, numNeighbors,
                                        query);

    for(i = 0;i < numNeighbors; i++){
        printf("%d %f\n",
               neighbors[i].index + 1, neighbors[i].distance);

        /* check distance */
        if(fabs(
                dist_l2(dim, query, &data[neighbors[i].index * dim]) -
                neighbors[i].distance) > 1e-6){
            printf("%d distance is different. %f\n",
                  dist_l2(dim, query, &data[neighbors[i].index * dim]) );
        }
        /* check order */
        if(i != 0 && neighbors[i-1].distance > neighbors[i].distance){
            printf("order is wrong.\n");
        }
    }

    vl_free(neighbors);

    vl_kdforest_delete(forest);
    free(data);
    free(query);
}
void
test_dist()
{
    void (*fdist)(void) = (void(*)(void))
        vl_get_vector_comparison_function_f (VlDistanceL2);
    
    float x[]={ 0, 0 };
    float y[]={ 2, 0 };

    printf("%f\n",
           ((VlFloatVectorComparisonFunction)fdist)(2, x, y));
}

int
main (int argc VL_UNUSED, char *argv[] VL_UNUSED)
{
    //test_dist();
    test_simple(1);
    test_query_from_file(1);

    return 0;
}
