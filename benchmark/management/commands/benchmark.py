import asyncio

from django.core.management.base import BaseCommand, CommandError

# Import the entry point functions from benchmark2/main.py
from benchmark2.main import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODELS,
    collect_results,
    create_benchmark_run,
    evaluate,
)


class Command(BaseCommand):
    help = "Benchmark different LLMs for code translation tasks"

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # translate command
        translate_parser = subparsers.add_parser(
            "translate", help="Translate code from one programming language to another"
        )
        translate_parser.add_argument(
            "--models",
            "-m",
            nargs="+",
            choices=SUPPORTED_MODELS,
            required=True,
            help="List of LLM models to benchmark",
        )
        translate_parser.add_argument(
            "--source",
            "-s",
            choices=SUPPORTED_LANGUAGES,
            default="python",
            help="Source programming language (default: python)",
        )
        translate_parser.add_argument(
            "--target",
            "-t",
            choices=SUPPORTED_LANGUAGES,
            default="go",
            help="Target programming language (default: go)",
        )
        translate_parser.add_argument(
            "--num-samples",
            "-n",
            type=int,
            default=None,
            help="Number of samples to process (default: all available)",
        )
        translate_parser.add_argument(
            "--parallel",
            "-p",
            action="store_true",
            default=False,
            help="Process samples and models in parallel (default: serial)",
        )

        # collect command
        collect_parser = subparsers.add_parser(
            "collect", help="Collect and aggregate results from database"
        )
        collect_parser.add_argument(
            "--benchmark-id",
            "-b",
            type=int,
            required=True,
            help="Benchmark run ID to collect results for",
        )

    async def run_model_evaluation(
        self, model: str, options: dict, benchmark_run: dict
    ) -> None:
        """Run evaluation for a single model."""
        self.stdout.write(f"\nTranslating with {model}...")
        await evaluate(
            model=model,
            source_lang=options["source"],
            target_lang=options["target"],
            benchmark_run=benchmark_run,
            num_samples=options["num_samples"],
            parallel=options["parallel"],
        )

    def handle(self, *args, **options):
        command = options["command"]

        if command == "translate":
            self.stdout.write(f"Models selected: {', '.join(options['models'])}")
            self.stdout.write(f"Source language: {options['source']}")
            self.stdout.write(f"Target language: {options['target']}")
            self.stdout.write(
                f"Number of samples: {'all' if options['num_samples'] is None else options['num_samples']}"
            )
            self.stdout.write(
                f"Processing mode: {'parallel' if options['parallel'] else 'serial'}"
            )

            # Create a benchmark run for all models
            async def create_run():
                return await create_benchmark_run(
                    models=options["models"],
                    source_lang=options["source"],
                    target_lang=options["target"],
                )

            # HACK to run async function in synchronous context
            benchmark_run = asyncio.run(create_run())
            self.stdout.write(
                f"Created benchmark run with ID: {benchmark_run['id']} and name: {benchmark_run['name']} for models: {', '.join(benchmark_run['model_names'])}"
            )

            if options["parallel"] and len(options["models"]) > 1:
                # Run models in parallel when parallel flag is set and multiple models are specified
                self.stdout.write("Running multiple models in parallel...")

                async def run_all_models():
                    tasks = []
                    for model in options["models"]:
                        task = self.run_model_evaluation(model, options, benchmark_run)
                        tasks.append(task)
                    await asyncio.gather(*tasks)

                asyncio.run(run_all_models())
            else:
                # Run models serially
                for model in options["models"]:
                    self.stdout.write(f"\nTranslating with {model}...")
                    asyncio.run(
                        evaluate(
                            model=model,
                            source_lang=options["source"],
                            target_lang=options["target"],
                            benchmark_run=benchmark_run,
                            num_samples=options["num_samples"],
                            parallel=options["parallel"],
                        )
                    )

        elif command == "collect":
            benchmark_id = options["benchmark_id"]
            self.stdout.write(f"Collecting results for benchmark run {benchmark_id}")
            asyncio.run(collect_results(benchmark_id))

        else:
            raise CommandError("Please specify a valid command: translate or collect")
