#import dataset_prepare_cy
#dataset_prepare_cy.main()

import timeit

cy = timeit.timeit('dataset_prepare_cy.main()', setup='import dataset_prepare_cy', number=1)
py = timeit.timeit('dataset_prepare.main()', setup='import dataset_prepare', number=1)

print(cy, py)
print('Cython is {}x faster'.format(py/cy))
