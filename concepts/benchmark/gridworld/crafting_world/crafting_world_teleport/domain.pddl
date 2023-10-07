(define
 (domain crafting-world-v20230404-teleport)
 (:requirements :strips)
 (:types
   tile
   object
   inventory
   object-type
 )
 (:predicates
   ;; in this teleport environment, we don't need the map definition.
   ; (tile-up ?t1 - tile ?t2 - tile)     ;; t2 is up of t1
   ; (tile-down ?t1 - tile ?t2 - tile)   ;; t2 is down of t1
   ; (tile-left ?t1 - tile ?t2 - tile)   ;; t2 is left of t1
   ; (tile-right ?t1 - tile ?t2 - tile)  ;; t2 is right of t1

   (agent-at ?t - tile)
   (object-at ?x - object ?t - tile)
   (inventory-holding ?i - inventory ?x - object)
   (inventory-empty ?i - inventory)

   (object-of-type ?x - object ?ot - object-type)
 )
 (:constants
  Key - object-type
  WorkStation - object-type
  Pickaxe - object-type
  IronOreVein - object-type
  IronOre - object-type
  IronIngot - object-type
  CoalOreVein - object-type
  Coal - object-type
  GoldOreVein - object-type
  GoldOre - object-type
  GoldIngot - object-type
  CobblestoneStash - object-type
  Cobblestone - object-type
  Axe - object-type
  Tree - object-type
  Wood - object-type
  WoodPlank - object-type
  Stick - object-type
  WeaponStation - object-type
  Sword - object-type
  Chicken - object-type
  Feather - object-type
  Arrow - object-type
  ToolStation - object-type
  Shears - object-type
  Sheep - object-type
  Wool - object-type
  Bed - object-type
  BedStation - object-type
  BoatStation - object-type
  Boat - object-type
  SugarCanePlant - object-type
  SugarCane - object-type
  Paper - object-type
  Furnace - object-type
  FoodStation - object-type
  Bowl - object-type
  PotatoPlant - object-type
  Potato - object-type
  CookedPotato - object-type
  BeetrootCrop - object-type
  Beetroot - object-type
  BeetrootSoup - object-type

  Hypothetical - object-type
  Trash - object-type
 )
 (:action move-to
  :parameters (?t1 - tile ?t2 - tile)
  :precondition (and (agent-at ?t1))
  :effect (and (agent-at ?t2) (not (agent-at ?t1)))
 )
 (:action pick-up
  :parameters (?i - inventory ?x - object ?t - tile)
  :precondition (and (agent-at ?t) (object-at ?x ?t) (inventory-empty ?i))
  :effect (and (inventory-holding ?i ?x) (not (object-at ?x ?t)) (not (inventory-empty ?i)))
 )
 (:action place-down
  :parameters (?i - inventory ?x - object ?t - tile)
  :precondition (and (agent-at ?t) (inventory-holding ?i ?x))
  :effect (and (object-at ?x ?t) (not (inventory-holding ?i ?x)) (inventory-empty ?i))
 )

 (:regression move [always]
  :parameters ((forall ?t1 - tile) ?t2 - tile)
  :goal (agent-at ?t2)
  :precondition (and (agent-at ?t1))
  :rule (then
    (move-to ?t1 ?t2)
  )
 )
 (:regression pick-up [always]
  :parameters (?target-type - object-type (forall ?target-inventory - inventory) (forall ?target - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x ?target-type))))
  :precondition (and (object-at ?target ?t) (object-of-type ?target ?target-type) (inventory-empty ?target-inventory))
  :rule (then
    (achieve (agent-at ?t))
    (pick-up ?target-inventory ?target ?t)
  )
 )


 (:action mine-iron-ore
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x IronOreVein)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Pickaxe)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target IronOre)
  )
 )
 (:regression mine-iron-ore-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronOre))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource IronOreVein) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Pickaxe)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-iron-ore ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-iron-ore-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronOre))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Pickaxe)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronOre)))))
  )
 )
 
 (:action mine-coal
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x CoalOreVein)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Pickaxe)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Coal)
  )
 )
 (:regression mine-coal-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Coal))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource CoalOreVein) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Pickaxe)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-coal ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-coal-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Coal))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Pickaxe)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Coal)))))
  )
 )
 
 (:action mine-cobblestone
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x CobblestoneStash)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Pickaxe)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Cobblestone)
  )
 )
 (:regression mine-cobblestone-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Cobblestone))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource CobblestoneStash) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Pickaxe)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-cobblestone ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-cobblestone-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Cobblestone))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Pickaxe)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Cobblestone)))))
  )
 )
 
 (:action mine-wood
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x Tree)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Axe)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Wood)
  )
 )
 (:regression mine-wood-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wood))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource Tree) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Axe)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-wood ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-wood-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wood))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Axe)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wood)))))
  )
 )
 
 (:action mine-feather
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x Chicken)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Sword)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Feather)
  )
 )
 (:regression mine-feather-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Feather))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource Chicken) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Sword)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-feather ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-feather-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Feather))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Sword)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Feather)))))
  )
 )
 
 (:action mine-wool1
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x Sheep)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Shears)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Wool)
  )
 )
 (:regression mine-wool1-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wool))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource Sheep) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Shears)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-wool1 ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-wool1-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wool))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Shears)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wool)))))
  )
 )
 
 (:action mine-wool2
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x Sheep)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Sword)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Wool)
  )
 )
 (:regression mine-wool2-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wool))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource Sheep) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Sword)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-wool2 ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-wool2-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wool))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Sword)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wool)))))
  )
 )
 
 (:action mine-potato
  :parameters (?targetinv - inventory ?x - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x PotatoPlant)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Potato)
  )
 )
 (:regression mine-potato-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Potato))))
  :precondition (and (object-at ?target-resource ?t) (object-of-type ?target-resource PotatoPlant) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical))
  :rule (then
    (achieve (agent-at ?t))
    (mine-potato ?target-inventory ?target-resource ?target ?t)
  )
 )
 (:action mine-beetroot
  :parameters (?targetinv - inventory ?x - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x BeetrootCrop)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Beetroot)
  )
 )
 (:regression mine-beetroot-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Beetroot))))
  :precondition (and (object-at ?target-resource ?t) (object-of-type ?target-resource BeetrootCrop) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical))
  :rule (then
    (achieve (agent-at ?t))
    (mine-beetroot ?target-inventory ?target-resource ?target ?t)
  )
 )
 (:action mine-gold-ore
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x GoldOreVein)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Pickaxe)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target GoldOre)
  )
 )
 (:regression mine-gold-ore-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldOre))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource GoldOreVein) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding Pickaxe)
  )
  :rule (then
    (achieve (agent-at ?t))
    (mine-gold-ore ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression mine-gold-ore-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldOre))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Pickaxe)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldOre)))))
  )
 )
 
 (:action mine-sugar-cane
  :parameters (?targetinv - inventory ?x - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x SugarCanePlant)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target SugarCane)
  )
 )
 (:regression mine-sugar-cane-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x SugarCane))))
  :precondition (and (object-at ?target-resource ?t) (object-of-type ?target-resource SugarCanePlant) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical))
  :rule (then
    (achieve (agent-at ?t))
    (mine-sugar-cane ?target-inventory ?target-resource ?target ?t)
  )
 )

 (:action craft-wood-plank
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station WorkStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 Wood)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target WoodPlank)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 Wood))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-wood-plank-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x WoodPlank))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource WorkStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 Wood)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-wood-plank ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-wood-plank-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x WoodPlank))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wood)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x WoodPlank)))))
  )
 )
 
 (:action craft-stick
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station WorkStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 WoodPlank)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Stick)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 WoodPlank))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-stick-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Stick))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource WorkStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 WoodPlank)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-stick ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-stick-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Stick))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x WoodPlank)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Stick)))))
  )
 )
 
 (:action craft-arrow
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station WeaponStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 Feather)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Stick)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Arrow)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 Feather))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Stick))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression craft-arrow-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Arrow))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource WeaponStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 Feather)
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 Stick)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-arrow ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression craft-arrow-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Arrow))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Feather)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Stick)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Arrow)))))
  )
 )
 
 (:action craft-sword
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station WeaponStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 IronIngot)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Stick)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Sword)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 IronIngot))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Stick))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression craft-sword-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Sword))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource WeaponStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 IronIngot)
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 Stick)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-sword ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression craft-sword-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Sword))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronIngot)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Stick)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Sword)))))
  )
 )
 
 (:action craft-shears1
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station ToolStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 IronIngot)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Shears)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 IronIngot))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-shears1-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Shears))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource ToolStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 IronIngot)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-shears1 ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-shears1-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Shears))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronIngot)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Shears)))))
  )
 )
 
 (:action craft-shears2
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station ToolStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 GoldIngot)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Shears)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 GoldIngot))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-shears2-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Shears))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource ToolStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 GoldIngot)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-shears2 ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-shears2-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Shears))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldIngot)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Shears)))))
  )
 )
 
 (:action craft-iron-ingot
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station Furnace)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 IronOre)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Coal)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target IronIngot)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 IronOre))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Coal))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression craft-iron-ingot-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronIngot))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource Furnace) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 IronOre)
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 Coal)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-iron-ingot ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression craft-iron-ingot-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronIngot))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronOre)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Coal)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronIngot)))))
  )
 )
 
 (:action craft-gold-ingot
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station Furnace)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 GoldOre)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Coal)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target GoldIngot)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 GoldOre))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Coal))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression craft-gold-ingot-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldIngot))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource Furnace) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 GoldOre)
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 Coal)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-gold-ingot ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression craft-gold-ingot-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldIngot))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldOre)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Coal)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x GoldIngot)))))
  )
 )
 
 (:action craft-bed
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station BedStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 WoodPlank)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Wool)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Bed)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 WoodPlank))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Wool))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression craft-bed-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bed))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource BedStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 WoodPlank)
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 Wool)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-bed ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression craft-bed-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bed))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x WoodPlank)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Wool)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bed)))))
  )
 )
 
 (:action craft-boat
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station BoatStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 WoodPlank)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Boat)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 WoodPlank))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-boat-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Boat))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource BoatStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 WoodPlank)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-boat ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-boat-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Boat))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x WoodPlank)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Boat)))))
  )
 )
 
 (:action craft-bowl1
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station FoodStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 WoodPlank)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Bowl)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 WoodPlank))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-bowl1-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bowl))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource FoodStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 WoodPlank)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-bowl1 ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-bowl1-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bowl))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x WoodPlank)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bowl)))))
  )
 )
 
 (:action craft-bowl2
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station FoodStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 IronIngot)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Bowl)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 IronIngot))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-bowl2-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bowl))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource FoodStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 IronIngot)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-bowl2 ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-bowl2-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bowl))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x IronIngot)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bowl)))))
  )
 )
 
 (:action craft-cooked-potato
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station Furnace)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 Potato)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Coal)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target CookedPotato)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 Potato))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Coal))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression craft-cooked-potato-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x CookedPotato))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource Furnace) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 Potato)
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 Coal)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-cooked-potato ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression craft-cooked-potato-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x CookedPotato))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Potato)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Coal)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x CookedPotato)))))
  )
 )
 
 (:action craft-beetroot-soup
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station FoodStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 Beetroot)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Bowl)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target BeetrootSoup)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 Beetroot))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Bowl))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression craft-beetroot-soup-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x BeetrootSoup))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource FoodStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 Beetroot)
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 Bowl)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-beetroot-soup ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression craft-beetroot-soup-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x BeetrootSoup))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Beetroot)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Bowl)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x BeetrootSoup)))))
  )
 )
 
 (:action craft-paper
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station WorkStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 SugarCane)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Paper)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 SugarCane))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression craft-paper-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Paper))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource WorkStation) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 SugarCane)
  )
  :rule (then
    (achieve (agent-at ?t))
    (craft-paper ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression craft-paper-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Paper))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x SugarCane)))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x Paper)))))
  )
 )
 
)
