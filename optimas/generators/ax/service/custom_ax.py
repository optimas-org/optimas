"""Contains the definition of custom Ax classes."""

from typing import Optional, Tuple

from ax.service.ax_client import (
    AxClient,
    FixedFeatures,
    GeneratorRun,
    InstantiationBase,
    MaxParallelismReachedException,
    ObservationFeatures,
    OptimizationShouldStop,
    TParameterization,
    logger,
    round_floats_for_logging,
    manual_seed,
    not_none,
    retry_on_exception,
    CHOLESKY_ERROR_ANNOTATION,
)


class CustomAxClient(AxClient):
    """Custom AxClient that supports `fixed_features` in `gen_next_trial`.

    This class can be removed if https://github.com/facebook/Ax/pull/2015
    is merged.
    """

    @retry_on_exception(
        logger=logger,
        exception_types=(RuntimeError,),
        check_message_contains=["Cholesky", "cholesky"],
        suppress_all_errors=False,
        wrap_error_message_in=CHOLESKY_ERROR_ANNOTATION,
    )
    def get_next_trial(
        self,
        ttl_seconds: Optional[int] = None,
        force: bool = False,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> Tuple[TParameterization, int]:
        """Generate trial with the next set of parameters to try.

        This is a modified method that supports `fixed_features` as argument.

        Note: Service API currently supports only 1-arm trials.

        Parameters
        ----------
            ttl_seconds : int, optional
                If specified, will consider the trial failed after this
                many seconds. Used to detect dead trials that were not marked
                failed properly.
            force : bool, optional
                If set to True, this function will bypass the global stopping
                strategy's decision and generate a new trial anyway.
            fixed_features : ObservationFeatures, optional
                An ObservationFeatures object containing any
                features that should be fixed at specified values during
                generation.

        Returns
        -------
            Tuple of trial parameterization, trial index
        """
        # Check if the global stopping strategy suggests to stop the
        # optimization.
        # This is needed only if there is actually a stopping strategy
        # specified, and if this function is not forced to generate a new
        # trial.
        if self.global_stopping_strategy and (not force):
            # The strategy itself will check if enough trials have already been
            # completed.
            (
                stop_optimization,
                global_stopping_message,
            ) = self.global_stopping_strategy.should_stop_optimization(
                experiment=self.experiment
            )
            if stop_optimization:
                raise OptimizationShouldStop(message=global_stopping_message)

        try:
            trial = self.experiment.new_trial(
                generator_run=self._gen_new_generator_run(
                    fixed_features=fixed_features
                ),
                ttl_seconds=ttl_seconds,
            )
        except MaxParallelismReachedException as e:
            if self._early_stopping_strategy is not None:
                e.message += (  # noqa: B306
                    " When stopping trials early, make sure to call "
                    "`stop_trial_early` on the stopped trial."
                )
            raise e
        logger.info(
            f"Generated new trial {trial.index} with parameters "
            f"{round_floats_for_logging(item=not_none(trial.arm).parameters)}."
        )
        trial.mark_running(no_runner_required=True)
        self._save_or_update_trial_in_db_if_possible(
            experiment=self.experiment,
            trial=trial,
        )
        # TODO[T79183560]: Ensure correct handling of generator run when using
        # foreign keys.
        self._update_generation_strategy_in_db_if_possible(
            generation_strategy=self.generation_strategy,
            new_generator_runs=[self.generation_strategy._generator_runs[-1]],
        )
        return not_none(trial.arm).parameters, trial.index

    def _gen_new_generator_run(
        self, n: int = 1, fixed_features: Optional[ObservationFeatures] = None
    ) -> GeneratorRun:
        """Generate new generator run for this experiment.

        Parameters
        ----------
            n: int, optional
                Number of arms to generate.
            fixed_features: ObservationFeatures, optional,
                An ObservationFeatures object containing any
                features that should be fixed at specified values during
                generation.
        """
        # If random seed is not set for this optimization, context manager does
        # nothing; otherwise, it sets the random seed for torch, but only for
        # the scope of this call. This is important because torch seed is set
        # globally, so if we just set the seed without the context manager, it
        # can have serious negative impact on the performance of the models
        # that employ stochasticity.

        fixed_feats = InstantiationBase.make_fixed_observation_features(
            fixed_features=FixedFeatures(
                parameters={},
                trial_index=self._get_last_completed_trial_index(),
            )
        )
        if fixed_features:
            fixed_feats.update_features(fixed_features)
        with manual_seed(seed=self._random_seed):
            return not_none(self.generation_strategy).gen(
                experiment=self.experiment,
                n=n,
                pending_observations=self._get_pending_observation_features(
                    experiment=self.experiment
                ),
                fixed_features=fixed_feats,
            )
