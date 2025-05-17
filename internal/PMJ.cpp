#include "PMJ.h"

#include <cassert>
#include <random>

void Ray::UpdateStrata(const Ref::dvec2 &p, const int next_sample_count, const int dim,
                       std::vector<std::vector<bool>> &strata, std::vector<const Ref::dvec2 *> &sample_grid) {
    for (int i = 0, strata_cols = next_sample_count, strata_rows = 1; strata_cols >= 1;
         strata_cols /= 2, strata_rows *= 2, ++i) {
        const int x = int(p.get<0>() * strata_cols);
        const int y = int(p.get<1>() * strata_rows);
        strata[i][y * strata_cols + x] = true;
    }

    const int x = int(p.get<0>() * dim), y = int(p.get<1>() * dim);
    assert(!sample_grid[y * dim + x]);
    sample_grid[y * dim + x] = &p;
}

void Ray::GetValidXOffsets(const int x_pos, const int y_pos, const int strata_index,
                           const std::vector<std::vector<bool>> &strata, std::vector<int> &x_offsets) {
    const int strata_cols = 1 << (strata.size() - strata_index - 1);
    const bool is_occupied = strata[strata_index][y_pos * strata_cols + x_pos];
    if (is_occupied) {
        return;
    }
    if (strata_index == 0) {
        x_offsets.push_back(x_pos);
    } else {
        GetValidXOffsets(x_pos * 2, y_pos / 2, strata_index - 1, strata, x_offsets);
        GetValidXOffsets(x_pos * 2 + 1, y_pos / 2, strata_index - 1, strata, x_offsets);
    }
}

void Ray::GetValidYOffsets(const int x_pos, const int y_pos, const int strata_index,
                           const std::vector<std::vector<bool>> &strata, std::vector<int> &y_offsets) {
    const int strata_cols = 1 << (strata.size() - strata_index - 1);
    const bool is_occupied = strata[strata_index][y_pos * strata_cols + x_pos];
    if (is_occupied) {
        return;
    }
    if (strata_index == strata.size() - 1) {
        y_offsets.push_back(y_pos);
    } else {
        GetValidYOffsets(x_pos / 2, y_pos * 2, strata_index + 1, strata, y_offsets);
        GetValidYOffsets(x_pos / 2, y_pos * 2 + 1, strata_index + 1, strata, y_offsets);
    }
}

void Ray::GetValidOffsets(const int x_pos, const int y_pos, const std::vector<std::vector<bool>> &strata,
                          std::vector<int> &x_offsets, std::vector<int> &y_offsets) {
    if (strata.size() % 2 != 0) {
        GetValidXOffsets(x_pos, y_pos, int(strata.size() / 2), strata, x_offsets);
        GetValidYOffsets(x_pos, y_pos, int(strata.size() / 2), strata, y_offsets);
    } else {
        GetValidXOffsets(x_pos, y_pos / 2, int(strata.size() / 2 - 1), strata, x_offsets);
        GetValidYOffsets(x_pos / 2, y_pos, int(strata.size() / 2), strata, y_offsets);
    }
}

Ray::aligned_vector<Ray::Ref::dvec2> Ray::GeneratePMJSamples(const unsigned int seed, const int sample_count,
                                                             const int candidates_count) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> uniform_real;
    std::uniform_int_distribution<int> uniform_int;

    aligned_vector<Ref::dvec2> all_samples;
    std::vector<std::vector<bool>> strata{{false}};
    int next_sample_count = 1;
    int dim = 1;

    // avoid reallocation
    all_samples.reserve(sample_count);

    std::vector<const Ref::dvec2 *> sample_grid(sample_count, nullptr);

    auto subdivide_strata = [&]() {
        const int sample_count = next_sample_count;

        next_sample_count *= 2;
        const bool is_power_of_4 = (next_sample_count & 0x55555555) != 0;
        if (!is_power_of_4) {
            dim *= 2;
        }

        strata.resize(strata.size() + 1);

        std::fill(begin(strata), end(strata), std::vector<bool>(next_sample_count, false));
        std::fill(begin(sample_grid), begin(sample_grid) + next_sample_count, nullptr);
        for (int i = 0; i < sample_count; ++i) {
            UpdateStrata(all_samples[i], next_sample_count, dim, strata, sample_grid);
        }
    };

    auto get_sample_candidate = [&](const std::vector<int> &valid_offsets_x, const std::vector<int> &valid_offsets_y) {
        std::uniform_int_distribution<int>::param_type param_x(0, int(valid_offsets_x.size() - 1)),
            param_y(0, int(valid_offsets_y.size() - 1));

        const int off_x = valid_offsets_x[uniform_int(gen, param_x)];
        const int off_y = valid_offsets_y[uniform_int(gen, param_y)];

        const double strata_width = 1.0 / next_sample_count;

        std::uniform_real_distribution<double>::param_type dparam_x(strata_width * off_x, strata_width * (off_x + 1)),
            dparam_y(strata_width * off_y, strata_width * (off_y + 1));
        return Ref::dvec2{uniform_real(gen, dparam_x), uniform_real(gen, dparam_y)};
    };

    auto generate_new_sample = [&](const int sample_x_pos, const int sample_y_pos, const int candidates) {
        std::vector<int> valid_offsets_x, valid_offsets_y;
        GetValidOffsets(sample_x_pos, sample_y_pos, strata, valid_offsets_x, valid_offsets_y);

        if (candidates == 1) {
            return get_sample_candidate(valid_offsets_x, valid_offsets_y);
        } else {
            Ref::dvec2 best_candidate = {};
            double max_min_dist_sq = 0.0;
            for (int c = 0; c < candidates; ++c) {
                const Ref::dvec2 candidate = get_sample_candidate(valid_offsets_x, valid_offsets_y);

                const int x_pos = int(candidate.get<0>() * dim), y_pos = int(candidate.get<1>() * dim);

                double min_dist_sq = 2.0;
#if 1
                auto update_min_dist = [](const Ref::dvec2 &candidate,
                                          const std::vector<const Ref::dvec2 *> &sample_grid, const int x, const int y,
                                          const int dim, double *out_min_dist_sq) {
                    const int wrapped_x = (x + dim) % dim, wrapped_y = (y + dim) % dim;

                    const Ref::dvec2 *neighbour = sample_grid[wrapped_y * dim + wrapped_x];
                    if (neighbour) {
                        double diff_x = std::abs(neighbour->get<0>() - candidate.get<0>());
                        if (diff_x > 0.5) {
                            diff_x = 1.0 - diff_x;
                        }
                        double diff_y = std::abs(neighbour->get<1>() - candidate.get<1>());
                        if (diff_y > 0.5) {
                            diff_y = 1.0 - diff_y;
                        }

                        const double dist_sq = (diff_x * diff_x) + (diff_y * diff_y);
                        if (dist_sq < *out_min_dist_sq) {
                            *out_min_dist_sq = dist_sq;
                        }
                    }
                };

                const double grid_size = 1.0 / dim;
                for (int i = 1; i <= dim / 2; ++i) {
                    const int x_min = x_pos - i, x_max = x_pos + i;
                    const int y_min = y_pos - i, y_max = y_pos + i;

                    int x = x_min, y = y_min;
                    for (; x < x_max; ++x) {
                        update_min_dist(candidate, sample_grid, x, y, dim, &min_dist_sq);
                    }
                    for (; y < y_max; ++y) {
                        update_min_dist(candidate, sample_grid, x, y, dim, &min_dist_sq);
                    }
                    for (; x > x_min; --x) {
                        update_min_dist(candidate, sample_grid, x, y, dim, &min_dist_sq);
                    }
                    for (; y > y_min; --y) {
                        update_min_dist(candidate, sample_grid, x, y, dim, &min_dist_sq);
                    }

                    const double grid_radius = grid_size * i;
                    const double grid_radius_sq = grid_radius * grid_radius;
                    if (min_dist_sq < grid_radius_sq || min_dist_sq < max_min_dist_sq) {
                        // early exit
                        break;
                    }
                }
#else
                for (int y = 0; y < dim; ++y) {
                    for (int x = 0; x < dim; ++x) {
                        if (x == x_pos && y == y_pos) {
                            continue;
                        }

                        const int wrapped_x = (x + dim) % dim;
                        const int wrapped_y = (y + dim) % dim;

                        const pmj_point_t *neighbour = sample_grid[wrapped_y * dim + wrapped_x];
                        if (neighbour) {
                            double diff_x = std::abs(neighbour->x - candidate.x);
                            if (diff_x > 0.5) {
                                diff_x = 1.0 - diff_x;
                            }
                            double diff_y = std::abs(neighbour->y - candidate.y);
                            if (diff_y > 0.5) {
                                diff_y = 1.0 - diff_y;
                            }

                            const double dist_sq = (diff_x * diff_x) + (diff_y * diff_y);
                            if (dist_sq < min_dist_sq) {
                                min_dist_sq = dist_sq;
                            }
                        }
                    }
                }
#endif
                if (min_dist_sq > max_min_dist_sq) {
                    best_candidate = candidate;
                    max_min_dist_sq = min_dist_sq;
                }
            }
            return best_candidate;
        }
    };

    auto get_subquadrants = [&]() {
        const int quad_dim = dim / 2;
        const int n = quad_dim * quad_dim;

        std::bernoulli_distribution dist;
        const bool swap_x = dist(gen);

        std::vector<Ref::ivec2> subquadrants;

        for (int i = 0; i < n; ++i) {
            const Ref::dvec2 &p = all_samples[i];
            int x = int(p.get<0>() * dim), y = int(p.get<1>() * dim);

            if (swap_x) {
                x ^= 1;
            } else {
                y ^= 1;
            }

            subquadrants.emplace_back(Ref::ivec2{x, y});
        }

        return subquadrants;
    };

    { // Generate first sample
        std::uniform_real_distribution<double>::param_type param(0.0, 1.0);
        all_samples.push_back(Ref::dvec2{uniform_real(gen, param), uniform_real(gen, param)});
        UpdateStrata(all_samples.back(), next_sample_count, dim, strata, sample_grid);
    }

    int n = 1;
    while (n < sample_count) {
        subdivide_strata();

        // Diagonally opposite samples
        for (int i = 0; i < n; ++i) {
            const Ref::dvec2 &p = all_samples[i];
            const int x = int(p.get<0>() * dim), y = int(p.get<1>() * dim);

            all_samples.push_back(generate_new_sample(x ^ 1, y ^ 1, candidates_count));
            UpdateStrata(all_samples.back(), next_sample_count, dim, strata, sample_grid);
        }

        subdivide_strata();

        // Horizontally or vertically opposite samples
        const std::vector<Ref::ivec2> subquadrants = get_subquadrants();
        for (int i = 0; i < n; ++i) {
            all_samples.push_back(
                generate_new_sample(subquadrants[i].get<0>(), subquadrants[i].get<1>(), candidates_count));
            UpdateStrata(all_samples.back(), next_sample_count, dim, strata, sample_grid);
        }

        // Diagonally opposite samples to the ones above
        for (int i = 0; i < n; ++i) {
            all_samples.push_back(
                generate_new_sample(subquadrants[i][0] ^ 1, subquadrants[i][1] ^ 1, candidates_count));
            UpdateStrata(all_samples.back(), next_sample_count, dim, strata, sample_grid);
        }

        n *= 4;
    }

    return all_samples;
}