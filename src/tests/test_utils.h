#ifndef TEST_UTILS_H
#define TEST_UTILS_H 1

/*
 * Changes here will likely require a manual 'make clean'
 * in order for changes to be picked up
 */
#define MSG_START fprintf(stderr, "Starting %s\n", __func__)
#define MSG_STOP fprintf(stderr, "Finished %s\n", __func__)
#define COMMENT(X) if (PRINT) puts(X);

// Print info while running tests
#define PRINT 0

#endif
