(define (domain mini-behavior)
    (:requirements
      :strips
      :typing
    )

    (:types
        human robot phyobj location - object
    )

    (:predicates
        (at ?p - phyobj ?l - location)
        (is-working ?h - human)
        (is-waiting ?h - human)

        (type-bowl ?p - phyobj)
        (type-plate ?p - phyobj)
        (type-cup ?p - phyobj)

        (type-cupboard ?p - location)
        (type-table ?p - location)
        (type-sink ?p - location)

        (size-large ?p - phyobj)
        (size-medium ?p - phyobj)
        (size-small ?p - phyobj)

        (color-red ?p - phyobj)
        (color-green ?p - phyobj)
        (color-blue ?p - phyobj)

        (state-clean ?p - phyobj)
        (state-full ?p - phyobj)
        (state-used ?p - phyobj)

        (on ?p1 - phyobj ?p2 - phyobj)
        (clear ?p - phyobj)

        (is-goal1 ?h - human)
        (is-goal2 ?h - human)
        (is-goal3 ?h - human)
        (is-goal4 ?h - human)
    )

    (:action human-move
        :parameters (?h - human ?p - phyobj ?from - location ?to - location)
        :precondition (and (at ?p ?from) (is-working ?h) (clear ?p))
        :effect (and (at ?p ?to) (not (at ?p ?from)))
    )
    (:action robot-move
        :parameters (?h - human ?p - phyobj ?from - location ?to - location)
        :precondition (and (at ?p ?from) (is-waiting ?h) (clear ?p))
        :effect (and (at ?p ?to) (not (at ?p ?from)) (is-working ?h) (not (is-waiting ?h)))
    )

    (:action human-make-full
        :parameters (?h - human ?p - phyobj ?c - location)
        :precondition (and (at ?p ?c) (state-clean ?p) (is-working ?h) (clear ?p) (type-cupboard ?c))
        :effect (and (state-full ?p) (not (state-clean ?p)))
    )
    (:action human-make-used
        :parameters (?h - human ?p - phyobj ?t - location)
        :precondition (and (at ?p ?t) (state-full ?p) (is-working ?h) (clear ?p) (type-table ?t))
        :effect (and (state-used ?p) (not (state-full ?p)))
    )
    (:action human-make-clean
        :parameters (?h - human ?p - phyobj ?s - location)
        :precondition (and (at ?p ?s) (state-used ?p) (is-working ?h) (clear ?p) (type-sink ?s))
        :effect (and (state-clean ?p) (not (state-used ?p)))
    )

    (:action human-stack-on
        :parameters (?h - human ?p1 - phyobj ?p2 - phyobj ?l - location)
        :precondition (and (at ?p1 ?l) (at ?p2 ?l) (is-working ?h) (clear ?p1) (clear ?p2) (type-plate ?p2))
        :effect (and (on ?p1 ?p2) (not (clear ?p2)))
    )

    (:action human-take-down
        :parameters (?h - human ?p1 - phyobj ?p2 - phyobj ?l - location)
        :precondition (and (at ?p1 ?l) (at ?p2 ?l) (is-working ?h) (clear ?p1) (on ?p1 ?p2))
        :effect (and (clear ?p2) (not (on ?p1 ?p2)))
    )

    (:action reach-goal1
        :parameters (?h - human ?p1 - phyobj ?b1 - phyobj ?c1 - phyobj ?t - location)
        :precondition (and (type-plate ?p1) (type-bowl ?b1) (type-cup ?c1)
                            (at ?p1 ?t) (at ?b1 ?t) (at ?c1 ?t)
                            (size-medium ?p1) (size-medium ?b1) (size-medium ?c1)
                            (state-full ?b1) (state-full ?c1)
                            (is-working ?h)
                            (type-table ?t))
        :effect (and (is-goal1 ?h))
    )

    (:action reach-goal2
        :parameters (?h - human ?p1 - phyobj ?p2 - phyobj ?p3 - phyobj ?c - location)
        :precondition (and (type-plate ?p1) (type-plate ?p2) (type-plate ?p3)
                            (at ?p1 ?c) (at ?p2 ?c) (at ?p3 ?c)
                            (on ?p1 ?p2) (on ?p2 ?p3)
                            (state-clean ?p1) (state-clean ?p2) (state-clean ?p3)
                            (is-working ?h)
                            (type-cupboard ?c))
        :effect (and (is-goal2 ?h))
    )

    (:action reach-goal3
        :parameters (?h - human ?p1 - phyobj ?p2 - phyobj ?c1 - phyobj ?t - location)
        :precondition (and (type-plate ?p1) (type-plate ?p2) (type-cup ?c1)
                            (at ?p1 ?t) (at ?p2 ?t) (at ?c1 ?t)
                            (on ?c1 ?p2) (size-small ?p2)
                            (state-clean ?p1) (state-clean ?p2) (state-clean ?c1)
                            (is-working ?h)
                            (type-table ?t))
        :effect (and (is-goal3 ?h))
    )

    (:action reach-goal4
        :parameters (?h - human ?o1 - phyobj ?o2 - phyobj ?o3 - phyobj ?s - location)
        :precondition (and (at ?o1 ?s) (at ?o2 ?s) (at ?o1 ?s)
                            (color-red ?o1) (color-green ?o2) (color-blue ?o3)
                            (is-working ?h)
                            (type-sink ?s))
        :effect (and (is-goal4 ?h))
    )
)
