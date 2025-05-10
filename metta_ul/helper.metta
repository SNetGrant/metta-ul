!(import! &self metta_ul)


(:kv-values (-> Expression Expression))
(= 
    (kv-values $expr)
    (if (== $expr ()) ()
            (let* (
                (($k $v) (car-atom $expr))
                ($tail (cdr-atom $expr))
                ($tail-new (kv-values $tail))
                )
              (cons-atom $v $tail-new)
            )
    )
)
