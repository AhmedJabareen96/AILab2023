# fitness function
using Statistics

function fitness(individual::AbstractArray)
    target = collect("Hello, World!")
    score = 0
    for i in eachindex(target)
        if(individual[i] == target[i])
            score += 1  
        end
    end
    return score
end

# Genetic Algorithm
function genetic_algorithm(pop_size::Int64, num_genes::Int64, fitness_func::Function, max_generations::Int64)
    
    # initialize population
    population = [] 
    for i in 1:pop_size
        individual = [Char(rand(32:126)) for j in 1:num_genes]
        push!(population, individual)
    end
    
    # Evolve the population for a fixed number of max_generations
    for generation in 1:max_generations
        # Evaluate the fitness of each individual
        fitnesses = [fitness_func(individual) for individual in population]
        average = sum(fitnesses) / length(fitnesses)
        standard_dev = std(fitnesses)
        elite_size = Int(pop_size * 0.1)
        elite_indices = sortperm(fitnesses, rev=true)[1:elite_size]
        elites = [population[i] for i in elite_indices]
        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while length(offspring) < (pop_size - elite_size)
            parent1 = elites[rand(1:length(elites))]
            parent2 = elites[rand(1:length(elites))]
            child = [rand() < 0.5 ? parent1[i] : parent2[i] for i in 1:num_genes]
            push!(offspring, child)
        end
        population = vcat(elites, offspring)
    end
    # find individual with highest fitness
    best_individual = argmax(fitness_func, population)
    best_fitness = fitness_func(best_individual)
    if best_fitness == num_genes
        println("Time to get global optimum is: ")
    else
        println("Time to get local optimum is: ")
    end

    return best_individual, best_fitness
end

function main()
    best_individual, best_fitness = genetic_algorithm(100,13,fitness,16384) 
    best_individual = join(best_individual)
    println("best_individual: $(best_individual)")
    println("best_fitness: $(best_fitness)")
end

main()