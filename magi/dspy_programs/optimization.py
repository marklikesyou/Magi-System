"""DSPy optimization pipeline for the MAGI system."""

from __future__ import annotations

from typing import Any ,Callable ,List ,Optional ,Tuple

from .signatures import STUB_MODE

if not STUB_MODE :
    import dspy
    from dspy .teleprompt import (
    BootstrapFewShot ,
    BootstrapFewShotWithRandomSearch ,
    MIPROv2 ,
    SignatureOptimizer ,
    )

    from .evaluation import (
    comprehensive_judge_metric ,
    magi_optimization_metric ,
    )
    from .personas import MagiProgram

    class MAGIOptimizer :


        def __init__ (
        self ,
        retriever :Callable [[str ],str ],
        trainset :List [Any ],
        valset :Optional [List [Any ]]=None ,
        metric :Optional [Callable ]=None ,
        ):
            """Initialize the MAGI optimizer.

            Args:
                retriever: Function to retrieve context for queries
                trainset: Training examples
                valset: Validation examples (optional)
                metric: Custom metric function (defaults to composite_magi_metric)
            """
            self .retriever =retriever
            self .trainset =trainset
            self .valset =valset or []
            self .metric =metric or magi_optimization_metric
            self .base_program =MagiProgram (retriever )

        def optimize_with_bootstrap (
        self ,
        max_bootstrapped_demos :int =8 ,
        max_labeled_demos :int =8 ,
        max_errors :int =5 ,
        **kwargs
        )->MagiProgram :
            """Optimize using BootstrapFewShot.

            This method automatically generates few-shot examples
            by running the program and collecting successful traces.
            """
            print ("Optimizing MAGI with BootstrapFewShot...")

            optimizer =BootstrapFewShot (
            metric =self .metric ,
            max_bootstrapped_demos =max_bootstrapped_demos ,
            max_labeled_demos =max_labeled_demos ,
            max_errors =max_errors ,
            **kwargs
            )

            optimized =optimizer .compile (
            self .base_program ,
            trainset =self .trainset
            )

            print (f"Optimization complete. Generated {len (optimizer .demos )} demonstrations.")
            return optimized

        def optimize_with_random_search (
        self ,
        num_candidates :int =10 ,
        max_bootstrapped_demos :int =8 ,
        max_labeled_demos :int =8 ,
        **kwargs
        )->MagiProgram :
            """Optimize using BootstrapFewShotWithRandomSearch.

            This explores multiple prompt variations and selects the best.
            """
            print (f"Optimizing MAGI with Random Search ({num_candidates } candidates)...")

            optimizer =BootstrapFewShotWithRandomSearch (
            metric =self .metric ,
            num_candidates =num_candidates ,
            max_bootstrapped_demos =max_bootstrapped_demos ,
            max_labeled_demos =max_labeled_demos ,
            **kwargs
            )

            optimized =optimizer .compile (
            self .base_program ,
            trainset =self .trainset ,
            valset =self .valset if self .valset else None
            )

            print (f"Optimization complete. Best score: {optimizer .best_score :.3f}")
            return optimized

        def optimize_with_mipro (
        self ,
        num_candidates :int =10 ,
        init_temperature :float =1.0 ,
        **kwargs
        )->MagiProgram :
            """Optimize using MIPROv2.

            Advanced optimization that jointly optimizes instructions
            and few-shot examples.
            """
            print ("Optimizing MAGI with MIPROv2...")

            optimizer =MIPROv2 (
            metric =self .metric ,
            num_candidates =num_candidates ,
            init_temperature =init_temperature ,
            **kwargs
            )

            optimized =optimizer .compile (
            self .base_program ,
            trainset =self .trainset ,
            valset =self .valset ,
            requires_permission_to_run =False
            )

            print ("MIPROv2 optimization complete.")
            return optimized

        def optimize_signatures (
        self ,
        prompt_model :Optional [str ]=None ,
        **kwargs
        )->MagiProgram :
            """Optimize the signature prompts themselves.

            This method optimizes the persona descriptions and prompts.
            """
            print ("Optimizing MAGI signatures...")

            optimizer =SignatureOptimizer (
            metric =self .metric ,
            prompt_model =prompt_model ,
            **kwargs
            )

            optimized =optimizer .compile (
            self .base_program ,
            trainset =self .trainset ,
            valset =self .valset
            )

            print ("Signature optimization complete.")
            return optimized

        def evaluate_optimization (
        self ,
        optimized_program :MagiProgram ,
        testset :Optional [List [Any ]]=None
        )->dict :
            """Evaluate the optimized program against the base program."""
            from .evaluation import create_magi_evaluator

            test_data =testset or self .valset or self .trainset [-10 :]

            print ("\nEvaluating base program...")
            base_evaluator =create_magi_evaluator (
            test_data ,
            metric =comprehensive_judge_metric
            )
            base_scores =base_evaluator (self .base_program )

            print ("\nEvaluating optimized program...")
            opt_evaluator =create_magi_evaluator (
            test_data ,
            metric =comprehensive_judge_metric
            )
            opt_scores =opt_evaluator (optimized_program )


            improvements ={}
            for key in base_scores :
                if isinstance (base_scores [key ],dict ):
                    improvements [key ]={
                    k :opt_scores [key ][k ]-base_scores [key ][k ]
                    for k in base_scores [key ]
                    }
                else :
                    improvements [key ]=opt_scores [key ]-base_scores [key ]

            print ("\n=== Optimization Results ===")
            print (f"Base Overall Score: {base_scores .get ('overall',0 ):.3f}")
            print (f"Optimized Overall Score: {opt_scores .get ('overall',0 ):.3f}")
            print (f"Improvement: {improvements .get ('overall',0 ):.3f}")

            return {
            "base":base_scores ,
            "optimized":opt_scores ,
            "improvements":improvements
            }

    class AdaptiveMAGI (dspy .Module ):
        def __init__ (
        self ,
        retriever :Callable [[str ],str ],
        feedback_judge :Optional [dspy .Signature ]=None ,
        max_history_size :int =50 ,
        adaptation_threshold :int =10
        ):
            super ().__init__ ()
            self .retriever =retriever
            self .base_program =MagiProgram (retriever )
            self .feedback_history =[]
            self .max_history_size =max_history_size
            self .adaptation_threshold =adaptation_threshold
            self .adaptation_count =0

            if feedback_judge :
                self .judge =dspy .Predict (feedback_judge )
            else :
                from .evaluation import MAGISystemJudge
                self .judge =dspy .Predict (MAGISystemJudge )

        def forward (
        self ,
        query :str ,
        constraints :str ="",
        collect_feedback :bool =True
        )->Tuple [Any ,dict ]:



            result ,personas =self .base_program (query ,constraints )

            if collect_feedback :
                feedback =self ._collect_feedback (
                query ,constraints ,result ,personas
                )
                self .feedback_history .append ({
                "query":query ,
                "result":result ,
                "feedback":feedback
                })


                if len (self .feedback_history )>=self .adaptation_threshold :
                    self ._adapt_from_feedback ()

            return result ,personas

        def _collect_feedback (
        self ,
        query :str ,
        constraints :str ,
        result :Any ,
        personas :dict
        )->dict :
            melchior =personas .get ("melchior",{})
            balthasar =personas .get ("balthasar",{})
            casper =personas .get ("casper",{})

            feedback =self .judge (
            query =query ,
            context =self .retriever (query ),
            melchior_output =str (melchior .get ("analysis","")),
            melchior_confidence =str (melchior .get ("confidence",0 )),
            balthasar_output =str (balthasar .get ("plan","")),
            balthasar_confidence =str (balthasar .get ("confidence",0 )),
            casper_output =f"Risks: {casper .get ('risks','')}\nMitigations: {casper .get ('mitigations','')}",
            casper_confidence =str (casper .get ("confidence",0 )),
            final_verdict =result .verdict ,
            final_justification =result .justification
            )

            return {
            "scientific_rigor":float (feedback .scientific_rigor ),
            "strategic_feasibility":float (feedback .strategic_feasibility ),
            "risk_management":float (feedback .risk_management ),
            "decision_quality":float (feedback .decision_quality ),
            "system_coherence":float (feedback .system_coherence ),
            "strengths":feedback .strengths ,
            "weaknesses":feedback .weaknesses ,
            "recommendation":feedback .recommendation
            }

        def _adapt_from_feedback (self ):
            trainset =[]
            for entry in self .feedback_history [-self .adaptation_threshold :]:
                if entry ["feedback"]["decision_quality"]>=0.7 :
                    trainset .append (entry )

            if len (trainset )>=5 :
                print (f"Adapting MAGI based on {len (trainset )} feedback examples...")
                optimizer =BootstrapFewShot (
                metric =lambda g ,p ,t :g .get ("feedback",{}).get ("decision_quality",0 )>=0.7 if isinstance (g ,dict )else getattr (p ,"decision_quality",0 )>=0.7 ,
                max_bootstrapped_demos =4 ,
                max_labeled_demos =4
                )

                try :
                    self .base_program =optimizer .compile (
                    self .base_program ,
                    trainset =trainset
                    )
                    print ("Adaptation complete.")
                    self .adaptation_count +=1
                except Exception as e :
                    print (f"Adaptation failed: {e }")

                if len (self .feedback_history )>self .max_history_size :
                    self .feedback_history =self .feedback_history [-self .max_history_size :]

    def create_optimization_pipeline (
    retriever :Callable [[str ],str ],
    trainset :List [Any ],
    valset :Optional [List [Any ]]=None ,
    optimization_method :str ="bootstrap",
    **kwargs
    )->MagiProgram :
        """Convenience function to create and run optimization pipeline.

        Args:
            retriever: Function to retrieve context
            trainset: Training examples
            valset: Validation examples (optional)
            optimization_method: One of "bootstrap", "random_search", "mipro", "signature"
            **kwargs: Additional arguments for the optimizer

        Returns:
            Optimized MagiProgram
        """
        optimizer =MAGIOptimizer (retriever ,trainset ,valset )

        if optimization_method =="bootstrap":
            return optimizer .optimize_with_bootstrap (**kwargs )
        elif optimization_method =="random_search":
            return optimizer .optimize_with_random_search (**kwargs )
        elif optimization_method =="mipro":
            return optimizer .optimize_with_mipro (**kwargs )
        elif optimization_method =="signature":
            return optimizer .optimize_signatures (**kwargs )
        else :
            raise ValueError (f"Unknown optimization method: {optimization_method }")

else :
    class MAGIOptimizer :
        def __init__ (self ,*args ,**kwargs ):
            raise NotImplementedError ("dspy missing")

    class AdaptiveMAGI :
        def __init__ (self ,*args ,**kwargs ):
            raise NotImplementedError ("dspy missing")

    def create_optimization_pipeline (*args ,**kwargs ):
        raise NotImplementedError ("dspy missing")


__all__ =[
"MAGIOptimizer",
"AdaptiveMAGI",
"create_optimization_pipeline",
]
