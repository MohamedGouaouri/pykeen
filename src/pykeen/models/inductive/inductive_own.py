# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Any, ClassVar, Mapping, Optional
from ...typing import Initializer

from class_resolver import Hint, HintOrType, OptionalKwargs

from .base import InductiveERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import (
    DistMultInteraction,
    Interaction,
    representation_resolver,
    NodePieceRepresentation
)

from ...nn.init import xavier_uniform_, xavier_normal_norm_
from ...triples.triples_factory import CoreTriplesFactory

__all__ = [
    "InductiveOwn",
]

logger = logging.getLogger(__name__)


class InductiveOwn(InductiveERModel):

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        inference_factory: CoreTriplesFactory,
        embedding_dim: int = 64,
        relation_representations_kwargs: OptionalKwargs = None,
        interaction: HintOrType[Interaction] = DistMultInteraction,
        validation_factory: Optional[CoreTriplesFactory] = None,
        test_factory: Optional[CoreTriplesFactory] = None,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_normal_norm_,

        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory of training triples. Must have create_inverse_triples set to True.
        :param inference_factory:
            the triples factory of inference triples. Must have create_inverse_triples set to True.
        :param validation_factory:
            the triples factory of validation triples. Must have create_inverse_triples set to True.
        :param test_factory:
            the triples factory of testing triples. Must have create_inverse_triples set to True.
        :param embedding_dim:
            the embedding dimension. Only used if embedding_specification is not given.
        :param relation_representations_kwargs:
            the relation representation parameters
        :param interaction:
            the interaction module, or a hint for it.

        :param kwargs:
            additional keyword-based arguments passed to :meth:`ERModel.__init__`

        :raises ValueError:
            if the triples factory does not create inverse triples
        """
        if not triples_factory.create_inverse_triples:
            raise ValueError(
                "The provided triples factory does not create inverse triples. However, for the node piece "
                "representations inverse relation representations are required.",
            )

        er = representation_resolver.make(
            # Make Embedding object
            query=None,
            pos_kwargs=relation_representations_kwargs,
            max_id=triples_factory.num_entities,
            shape=embedding_dim,
            # This might thow dimension exception
            # TODO: Migrate this to InductiveOwn
            initializer=entity_initializer
        )
        if validation_factory is None:
            validation_factory = inference_factory

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=NodePieceRepresentation,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                # triples_factory=triples_factory,
                # Modification here
                token_representations=er,
                initializer=entity_initializer
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
            ),
            validation_factory=validation_factory,
            testing_factory=test_factory,
            **kwargs,
        )
