from typing import Optional
from class_resolver import OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs
from ...nn import Representation
from ...nn import (
    representation_resolver,
)
from .base import InductiveERModel
from ...triples.triples_factory import CoreTriplesFactory


class InductiveOwnGNN(InductiveERModel):
    """
        Inductive model with GNN encoder
    """

    def __init__(self, *,
                 triples_factory: CoreTriplesFactory,
                 inference_factory: CoreTriplesFactory,
                 relation_representations_kwargs: OptionalKwargs = None,

                 validation_factory: Optional[CoreTriplesFactory] = None,
                 test_factory: Optional[CoreTriplesFactory] = None,
                 **kwargs) -> None:

        # super().__init__(
        #     triples_factory=triples_factory,
        #     entity_representations=entity_representations,
        #     entity_representations_kwargs=entity_representations_kwargs,
        #     validation_factory=validation_factory,
        #     testing_factory=testing_factory,
        #     **kwargs
        # )
        # relation_representations = representation_resolver.make(
        #     query=None,
        #     pos_kwargs=relation_representations_kwargs,
        #     max_id=2 * triples_factory.real_num_relations + 1,
        #     shape=embedding_dim,
        # )
        pass
